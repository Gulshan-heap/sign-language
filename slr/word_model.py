"""Keypoint-sequence word classifier (shared by ISL and ASL): preprocessing, model, training."""
import json
import numpy as np
import torch
import torch.nn as nn

from .landmarks import N_POINTS, LH, RH, FLIP_PERM

T_FIXED = 32
FEAT_DIM = N_POINTS * 2 * 2 + 2   # xy + frame-delta + 2 hand-present flags


# ---------------- preprocessing ----------------

def trim_and_resample(seq, t=T_FIXED):
    """(T, 67, 2) with NaN -> (t, 67, 2). Crops to frames where a hand is visible."""
    seq = seq.astype(np.float32)
    hand_vis = ~np.isnan(seq[:, LH.start:RH.stop, 0]).all(1)
    idx = np.where(hand_vis)[0]
    if len(idx) >= 4:
        seq = seq[idx[0]: idx[-1] + 1]
    if len(seq) == 0:
        return np.full((t, N_POINTS, 2), np.nan, np.float32)
    pick = np.linspace(0, len(seq) - 1, t).round().astype(int)
    return seq[pick]


def normalise(seq):
    """Centre on shoulder midpoint, scale by shoulder width. (t,67,2) -> same shape, NaN kept."""
    sh = seq[:, 11:13, :]
    mid = np.nanmedian(sh.mean(1), axis=0) if not np.isnan(sh).all() else np.array([0.5, 0.5])
    width = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)) if not np.isnan(sh).all() else 0.25
    if not np.isfinite(width) or width < 1e-3:
        width = 0.25
    return (seq - mid) / width


def to_features(seq):
    """(t, 67, 2) normalised (NaN allowed) -> (t, FEAT_DIM)."""
    present = np.stack([~np.isnan(seq[:, LH, 0]).all(1), ~np.isnan(seq[:, RH, 0]).all(1)], 1).astype(np.float32)
    x = np.nan_to_num(seq, nan=0.0)
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return np.concatenate([x.reshape(len(x), -1), d.reshape(len(x), -1) * 3, present], 1).astype(np.float32)


def preprocess(seq, t=T_FIXED):
    return to_features(normalise(trim_and_resample(seq, t)))


def augment(seq, rng):
    """Augment a raw (T, 67, 2) sequence (before preprocess)."""
    seq = seq.astype(np.float32).copy()
    if rng.random() < 0.3:                                  # mirror (left-handed signers / mirrored webcam)
        seq[..., 0] = 1.0 - seq[..., 0]
        seq = seq[:, FLIP_PERM]
    c = np.nanmean(seq.reshape(-1, 2), axis=0) if not np.isnan(seq).all() else np.zeros(2)
    ang = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    seq = (seq - c) @ rot.T * rng.uniform(0.85, 1.15) + c + rng.normal(0, 0.02, 2)
    seq = seq + rng.normal(0, 0.004, seq.shape)
    drop = rng.random(len(seq)) < 0.08                      # random dropped frames
    seq[drop] = np.nan
    if len(seq) > 8:                                        # random temporal crop
        n = len(seq)
        a, b = rng.integers(0, max(1, n // 8) + 1), n - rng.integers(0, max(1, n // 8) + 1)
        seq = seq[a:b]
    return seq


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
