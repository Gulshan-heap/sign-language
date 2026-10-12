"""Live decoding: turn a stream of per-frame landmarks into words (or letters).

WordSegmenter   (67, 2) keypoint frames -> one word per sign. A sign starts when a hand appears and ends when the
                hands have been gone for `gap` seconds (or after `max_s`); the collected clip goes to the word model.
                In manual mode the caller starts and stops the clip instead.
LetterStabiliser  per-frame letter guesses -> a letter is typed once it has been held steady for `hold` frames.

Both are plain Python/numpy (no camera, no MediaPipe), so they are easy to test with synthetic frames.
"""
import collections
import time

import numpy as np

from .landmarks import LH, RH

MIN_HAND_FRAMES = 6          # a clip needs at least this many frames with a hand to be worth predicting


def hands_visible(frame):
    """(67, 2) frame -> True if at least one hand was detected."""
    return not np.isnan(frame[LH.start:RH.stop, 0]).all()


class WordSegmenter:
    def __init__(self, predict, threshold=0.5, gap=0.7, max_s=6.0, start_frames=3, preroll=3):
        """predict(seq (T,67,2)) -> [(label, prob), ...] best first."""
        self.predict, self.threshold, self.gap, self.max_s = predict, threshold, gap, max_s
        self.start_frames, self.preroll = start_frames, preroll
        self.reset()

    def reset(self):
        self.state = "idle"                        # idle | signing
        self.buf, self.pre, self.seen, self.last_hand_t, self.t0 = [], collections.deque(maxlen=self.preroll), 0, 0.0, 0.0
        self.manual = False

    # --- manual control (hold-to-sign) ---
    def start(self, now=None):
        self.manual, self.state, self.buf, self.t0 = True, "signing", [], now or time.time()
        self.last_hand_t = self.t0

    def stop(self):
        self.manual = False
        return self._finish() if self.state == "signing" else None

    # --- per frame ---
    def push(self, frame, now=None):
        """-> None, or an event dict {accepted, top:[{label,prob}], frames, seconds} when a clip has just been decoded."""
        now = time.time() if now is None else now
        vis = hands_visible(frame)
        if self.state == "idle":
            if self.manual:
                return None
            self.pre.append(frame)
            self.seen = self.seen + 1 if vis else 0
            if self.seen >= self.start_frames:
                self.state, self.buf, self.t0, self.last_hand_t = "signing", list(self.pre), now, now
            return None
        self.buf.append(frame)
        if vis:
            self.last_hand_t = now
        if self.manual:
            return None
        if now - self.last_hand_t > self.gap or now - self.t0 > self.max_s:
            return self._finish()
        return None

    def _finish(self):
        buf, self.buf = self.buf, []
        self.state, self.seen = "idle", 0
        self.pre.clear()
        n_hand = sum(hands_visible(f) for f in buf)
        if n_hand < MIN_HAND_FRAMES:
            return {"accepted": False, "top": [], "frames": len(buf), "reason": "too short"}
        top = [{"label": l, "prob": float(p)} for l, p in self.predict(np.stack(buf))]
        return {"accepted": top[0]["prob"] >= self.threshold, "top": top, "frames": len(buf)}


class LetterStabiliser:
    """Type a letter once the same one has been predicted `hold` frames in a row (confidence >= threshold).

    The same letter is not typed again until the hand leaves, or `repeat` frames of continued holding have passed,
    so double letters ("ll") are possible without sticky keys.
    """

    def __init__(self, hold=8, threshold=0.6, repeat=24):
        self.hold, self.threshold, self.repeat = hold, threshold, repeat
        self.reset()

    def reset(self):
        self.cur, self.run, self.fired = None, 0, False

    def push(self, pred):
        """pred: (letter, prob) or None (no hand). -> (candidate letter or None, newly typed letter or None)"""
        if pred is None or pred[1] < self.threshold:
            self.cur, self.run, self.fired = None, 0, False
            return None, None
        letter = pred[0]
        if letter != self.cur:
            self.cur, self.run, self.fired = letter, 1, False
            return letter, None
        self.run += 1
        if not self.fired and self.run >= self.hold:
            self.fired = True
            return letter, letter
        if self.fired and self.run >= self.hold + self.repeat:
            self.run = self.hold
            return letter, letter
        return letter, None
