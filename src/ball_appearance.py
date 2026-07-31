"""
Ball appearance verification — stops the tracker locking onto HEADS.

Two cheap, video-robust signals computed on the candidate's pixels:

1. skin_ratio()  — heads (players AND crowd fans) are skin-coloured; a handball
   is not. A YCrCb skin gate is camera/lighting robust and needs no keypoints,
   so it catches the FAN heads in the stands that the pose-keypoint head-reject
   can't see. A blob that is mostly skin and not in a hand is a head, not a ball.

2. BallAppearance — an adaptive H-S colour histogram LEARNED from confident ball
   detections. New candidates are scored by similarity, so when a head-blob and
   the real ball both appear, the one that actually LOOKS like the ball wins.
   Neutral (0.5) until it has learned enough, so it never hurts a cold start.

Both are intentionally light (a few ms on ≤5 tiny crops per frame).
"""
from __future__ import annotations

import cv2
import numpy as np

# Classic YCrCb skin gate. Skin clusters tightly in Cr/Cb regardless of race or
# lighting; a yellow/white/blue ball sits well outside this box.
_CR_LO, _CR_HI = 133, 173
_CB_LO, _CB_HI = 77, 127
_Y_MIN = 40


def _clip_box(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def skin_ratio(frame, x1, y1, x2, y2) -> float:
    """Fraction of the candidate box that is skin-coloured (0..1)."""
    crop = _clip_box(frame, x1, y1, x2, y2)
    if crop is None:
        return 0.0
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    cr, cb, y = ycc[..., 1], ycc[..., 2], ycc[..., 0]
    mask = ((cr >= _CR_LO) & (cr <= _CR_HI) &
            (cb >= _CB_LO) & (cb <= _CB_HI) & (y >= _Y_MIN))
    return float(mask.mean())


class BallAppearance:
    """EMA H-S histogram of the ball, learned online from confident detections."""

    H_BINS, S_BINS = 12, 4
    EMA = 0.12             # slow adaptation → robust to one bad frame
    MIN_SAMPLES = 6        # stay neutral until we've actually seen the ball a few times

    def __init__(self):
        self._hist: np.ndarray | None = None
        self._n = 0

    def _hs_hist(self, frame, box):
        crop = _clip_box(frame, *box)
        if crop is None:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.H_BINS, self.S_BINS],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update(self, frame, box) -> None:
        h = self._hs_hist(frame, box)
        if h is None:
            return
        self._hist = h if self._hist is None else (1 - self.EMA) * self._hist + self.EMA * h
        self._n += 1

    @property
    def learned(self) -> bool:
        return self._hist is not None and self._n >= self.MIN_SAMPLES

    def similarity(self, frame, box) -> float:
        """Correlation with the learned ball colour, 0..1. Returns 0.5 (neutral)
        until enough samples are seen, so a cold start is never penalised."""
        if not self.learned:
            return 0.5
        h = self._hs_hist(frame, box)
        if h is None:
            return 0.5
        corr = cv2.compareHist(self._hist.astype("float32"),
                               h.astype("float32"), cv2.HISTCMP_CORREL)
        return float(max(0.0, min(1.0, (corr + 1.0) * 0.5)))   # [-1,1] → [0,1]
