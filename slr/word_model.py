"""Keypoint-sequence word classifier (shared by ISL and ASL): preprocessing, model, training."""
import json
import numpy as np
import torch
import torch.nn as nn


# preprocessing lives in features.py (numpy only); re-exported here for the training scripts
from .features import (T_FIXED, FEAT_DIM, trim_and_resample, normalise, to_features, preprocess, augment)  # noqa: E402,F401


# ---------------- model ----------------

class WordTransformer(nn.Module):
    def __init__(self, n_classes, d=128, layers=3, heads=4, drop=0.2, t=T_FIXED):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEAT_DIM, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, t, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        return self.head(h.mean(1))


def save_model(path, model, labels, extra=None):
    torch.save({"state": model.state_dict(), "labels": labels, "t": T_FIXED, "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = WordTransformer(len(ck["labels"]), t=ck["t"])
    m.load_state_dict(ck["state"])
    return m.eval().to(device), ck["labels"]


@torch.no_grad()
def predict(model, seq, device="cpu", topk=3):
    """seq: raw (T, 67, 2) keypoints -> list of (label_index, prob)."""
    x = torch.from_numpy(preprocess(seq))[None].to(device)
    p = torch.softmax(model(x), -1)[0].cpu().numpy()
    top = p.argsort()[::-1][:topk]
    return [(int(i), float(p[i])) for i in top]


# ---------------- training ----------------

def train(train_seqs, train_y, val_seqs, val_y, n_classes, epochs=150, lr=2e-3, batch=32,
          device="cpu", seed=0, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = WordTransformer(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * int(np.ceil(len(train_seqs) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_x = torch.from_numpy(np.stack([preprocess(s) for s in val_seqs])).to(device) if len(val_seqs) else None
    val_t = torch.tensor(val_y).to(device) if len(val_seqs) else None
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_seqs))
        tot = 0.0
        for i in range(0, len(order), batch):
            ids = order[i:i + batch]
            x = torch.from_numpy(np.stack([preprocess(augment(train_seqs[j], rng)) for j in ids])).to(device)
            y = torch.tensor([train_y[j] for j in ids]).to(device)
            loss = lossf(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(ids)
        if val_x is not None and (ep % 5 == 4 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                acc = (model(val_x).argmax(-1) == val_t).float().mean().item()
            if acc >= best:
                best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
            log(f"epoch {ep + 1:3d}  loss {tot / len(order):.3f}  val_acc {acc:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best


@torch.no_grad()
def evaluate(model, seqs, ys, device="cpu"):
    x = torch.from_numpy(np.stack([preprocess(s) for s in seqs])).to(device)
    logits = model(x)
    pred = logits.argmax(-1).cpu().numpy()
    top3 = logits.topk(min(3, logits.shape[1]), -1).indices.cpu().numpy()
    ys = np.asarray(ys)
    return {"top1": float((pred == ys).mean()), "top3": float(np.mean([y in t for y, t in zip(ys, top3)])),
            "n": int(len(ys))}, pred


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
