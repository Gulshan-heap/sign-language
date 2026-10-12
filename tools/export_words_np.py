"""Convert trained word models to numpy weights so the server does not need torch.

  python tools/export_words_np.py models          # models/asl_words.pt -> models/asl_words_np.npz (and isl)
Needs torch (run it on your laptop / Kaggle once after training). Checks that numpy and torch give the same output.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slr.numpy_word_model import NumpyWordModel  # noqa: E402
from slr.word_model import load_model, predict  # noqa: E402


def main(d):
    d = Path(d)
    for pt in sorted(d.glob("*_words.pt")):
        model, labels = load_model(pt)
        out = pt.with_name(pt.stem + "_np.npz")
        np.savez_compressed(out, labels=np.array(labels), **{k: v.numpy() for k, v in model.state_dict().items()})
        nm, rng, worst = NumpyWordModel(out), np.random.default_rng(0), 0.0
        for _ in range(5):                                   # random keypoint clips: compare top-3 probabilities
            seq = rng.random((int(rng.integers(12, 60)), 67, 2)).astype(np.float32)
            a, b = predict(model, seq), nm.predict(seq)
            assert [i for i, _ in a] == [i for i, _ in b], (a, b)
            worst = max(worst, max(abs(x[1] - y[1]) for x, y in zip(a, b)))
        print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  max prob difference vs torch: {worst:.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models")
