"""MediaPipe landmark helpers (MediaPipe *Tasks* API: works on current mediapipe, Kaggle's Python 3.13 and a Raspberry Pi).

Word models use one compact "frame" layout, (67, 2) = x, y per point:
    [0:25]  pose landmarks 0..24 (face outline, shoulders, arms, hips)
    [25:46] left hand (21)
    [46:67] right hand (21)
Missing parts are NaN.

The two .task model files (hand + pose, ~14 MB) are found in $SLR_MODEL_DIR, ./models, ./models/mp or ~/.cache/slr,
and downloaded into ~/.cache/slr when missing (needs Internet once; copy them to the Pi to run offline).
"""
import os
import urllib.request
from pathlib import Path

import numpy as np

N_POSE = 25
N_POINTS = N_POSE + 42
LH = slice(N_POSE, N_POSE + 21)
RH = slice(N_POSE + 21, N_POSE + 42)

# index permutation that mirrors a frame left <-> right
_POSE_SWAP = {1: 4, 2: 5, 3: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16, 17: 18, 19: 20, 21: 22, 23: 24}
FLIP_PERM = np.arange(N_POINTS)
for a, b in _POSE_SWAP.items():
    FLIP_PERM[a], FLIP_PERM[b] = b, a
FLIP_PERM[LH], FLIP_PERM[RH] = np.arange(RH.start, RH.stop), np.arange(LH.start, LH.stop)

_BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "hand_landmarker.task": _BASE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": _BASE + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def model_path(name):
    root = Path(__file__).resolve().parents[1]
    dirs = [os.environ.get("SLR_MODEL_DIR"), root / "models", root / "models" / "mp", Path.home() / ".cache" / "slr"]
    for d in dirs:
        if d and (Path(d) / name).exists():
            return str(Path(d) / name)
    dst = Path.home() / ".cache" / "slr" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(MODELS[name], dst)
    return str(dst)


def _mp_image(rgb):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class HandDetector:
    """rgb image -> list of (21, 3) arrays (x, y normalised to the image, z relative), best hand first.

    video=True enables tracking between consecutive calls (faster, use for webcam/video);
    video=False treats every call as an independent photo.
    """

    def __init__(self, num_hands=2, video=False):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.video, self.t = video, 0
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.det = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb):
        """-> (list of (21,3) arrays, list of 'Left'/'Right' labels)"""
        img = _mp_image(rgb)
        if self.video:
            self.t += 33
            res = self.det.detect_for_video(img, self.t)
        else:
            res = self.det.detect(img)
        hands = [np.array([[p.x, p.y, p.z] for p in h], np.float32) for h in res.hand_landmarks]
        labels = [c[0].category_name for c in res.handedness]
        return hands, labels

    def __call__(self, rgb):
        return self.detect(rgb)[0]

    def close(self):
        self.det.close()


class FrameTracker:
    """rgb video frame -> (67, 2) float32 in the layout above (NaN where not detected). Pose + hands."""

    def __init__(self):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.t = 0
        self.hands = HandDetector(2, video=True)
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))

    def __call__(self, rgb):
        frame = np.full((N_POINTS, 2), np.nan, np.float32)
        self.t += 33
        res = self.pose.detect_for_video(_mp_image(rgb), self.t)
        if res.pose_landmarks:
            frame[:N_POSE] = [[p.x, p.y] for p in res.pose_landmarks[0][:N_POSE]]
        hands, labels = self.hands.detect(rgb)
        used = set()
        for h, lab in zip(hands, labels):          # slot by handedness label; a 2nd hand with the same label takes the free slot
            slot = LH if lab == "Left" else RH
            if (slot.start in used):
                slot = RH if slot is LH else LH
            if slot.start in used:
                continue
            used.add(slot.start)
            frame[slot] = h[:, :2]
        return frame

    def close(self):
        self.hands.close()
        self.pose.close()


def video_to_keypoints(path, stride=1, max_frames=400, mirror=False):
    """Run the tracker over a video file -> (T, 67, 2) float16. `mirror` flips frames first."""
    import cv2
    tracker = FrameTracker()
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if mirror:
                img = img[:, ::-1]
            if max(img.shape[:2]) > 960:                      # 1080p -> 960 wide: same accuracy, much faster
                s = 960 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            frames.append(tracker(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    tracker.close()
    if not frames:
        return np.full((0, N_POINTS, 2), np.nan, np.float16)
    return np.stack(frames).astype(np.float16)


# ---------- single-image hand features (letters) ----------

def _norm_hand(h):
    """(21, 3) -> wrist-centred, scale-normalised flat vector."""
    h = h - h[0]
    s = np.abs(h[:, :2]).max()
    return (h / (s if s > 1e-6 else 1.0)).ravel()


def hands_to_features(hand_lists, n_hands):
    """List of (21,3) arrays -> fixed feature vector, or None if no hand found.

    Hands are ordered left-to-right in the image so the result does not depend on
    MediaPipe's handedness label (which flips with a mirrored webcam image).
    n_hands=1 -> 63 features; n_hands=2 -> 2*63 + 3 (relative wrist offset), zero padded.
    """
    if not hand_lists:
        return None
    hand_lists = sorted(hand_lists, key=lambda h: h[0, 0])[:n_hands]
    parts = [_norm_hand(h) for h in hand_lists]
    if n_hands == 1:
        return parts[0].astype(np.float32)
    parts += [np.zeros(63)] * (2 - len(parts))
    rel = (hand_lists[1][0] - hand_lists[0][0]) if len(hand_lists) == 2 else np.zeros(3)
    return np.concatenate(parts + [rel]).astype(np.float32)


def mirror_hands(hand_lists):
    """Mirror raw (21,3) hand landmarks horizontally (used as training augmentation)."""
    out = []
    for h in hand_lists:
        h = h.copy()
        h[:, 0] = 1.0 - h[:, 0]
        out.append(h)
    return out
