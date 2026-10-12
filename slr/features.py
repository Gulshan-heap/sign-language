"""Keypoint-sequence preprocessing for the word models. numpy only (no torch), so the server can run without it."""
import numpy as np

from .landmarks import N_POINTS, LH, RH, FLIP_PERM

T_FIXED = 32
FEAT_DIM = N_POINTS * 2 * 2 + 2   # xy + frame-delta + 2 hand-present flags


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
