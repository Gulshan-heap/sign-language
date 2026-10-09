"""MediaPipe landmark helpers.

Word models use one compact "frame" layout, (67, 2) = x, y per point:
    [0:25]  pose landmarks 0..24 (face outline, shoulders, arms, hips)
    [25:46] left hand (21)
    [46:67] right hand (21)
Missing parts are NaN.
"""
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


def _xy(landmark_list, n):
    if landmark_list is None:
        return np.full((n, 2), np.nan, np.float32)
    return np.array([[p.x, p.y] for p in landmark_list.landmark], np.float32)


def frame_from_holistic(res):
    """mediapipe Holistic result -> (67, 2) float32 (NaN where not detected)."""
    pose = _xy(res.pose_landmarks, 33)[:N_POSE]
    return np.concatenate([pose, _xy(res.left_hand_landmarks, 21), _xy(res.right_hand_landmarks, 21)])


def make_holistic(complexity=0):
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=complexity,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def video_to_keypoints(path, stride=1, complexity=0, max_frames=400, mirror=False):
    """Run Holistic over a video file -> (T, 67, 2) float16. `mirror` flips frames first."""
    import cv2
    holistic = make_holistic(complexity)
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if mirror:
                img = img[:, ::-1]
            res = holistic.process(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB))
            frames.append(frame_from_holistic(res))
        i += 1
    cap.release()
    holistic.close()
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


def make_hands(static=True, max_hands=1, complexity=1):
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=static, max_num_hands=max_hands, model_complexity=complexity,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def hands_from_result(res):
    if not res.multi_hand_landmarks:
        return []
    return [np.array([[p.x, p.y, p.z] for p in h.landmark], np.float32) for h in res.multi_hand_landmarks]
