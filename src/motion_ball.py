"""
Motion-Based Ball Candidate Generator.

Why this exists
---------------
A handball shot can travel >100 km/h. At 25 fps that's ≥1.1 m per frame, which
manifests as a heavily-blurred yellow streak. YOLO COCO's "sports ball" class
was trained mostly on white footballs and basketballs in clean, low-blur
photos — it routinely misses these motion-blurred handballs entirely. The user
keeps reporting "fast balls aren't detected" because of this fundamental
training-data mismatch.

This module provides a **second source** of ball candidates by running a cheap
motion-detection pass on each frame. Anything that:
    1. moved between this frame and the previous one,
    2. is roughly ball-sized (10..1500 px²),
    3. has a roundish bounding box (aspect 0.5..2.5),
    4. is NOT inside any tracked player's bbox (we don't want elbow trails)

becomes a synthetic ball candidate with a moderate confidence (0.30). The main
detector merges these with YOLO results — the existing tracker.update() picks
the best across both sources via its continuity / hand-magnet / body-penalty
scoring, so false positives still get filtered.

Cost
----
At 1280×720 the diff + threshold + connected-components costs ~3-5 ms per
frame on CPU (no GPU). Cheap enough to run every infer frame.

Usage
-----
    mb = MotionBall()
    cand_list = mb.find(frame_bgr, exclude_player_bboxes=[(x1,y1,x2,y2), ...])
    # cand_list: list[Detection-like] with .x1/.y1/.x2/.y2/.confidence/.class_id
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


BALL_CLASS = 32


@dataclass
class _MotionCandidate:
    """Duck-types Detection from detector.py."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = BALL_CLASS
    keypoints: Optional[np.ndarray] = None

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2


class MotionBall:
    def __init__(
        self,
        min_area_px:    int   = 12,
        max_area_px:    int   = 1500,
        aspect_max:     float = 2.5,    # motion blur can be very elongated
        diff_threshold: int   = 22,
        downscale:      float = 0.5,
        synth_conf:     float = 0.30,
        max_candidates: int   = 6,
    ):
        self.min_area    = int(min_area_px)
        self.max_area    = int(max_area_px)
        self.aspect_max  = float(aspect_max)
        self.diff_thr    = int(diff_threshold)
        self.downscale   = max(0.25, min(1.0, float(downscale)))
        self.synth_conf  = float(synth_conf)
        self.max_cands   = int(max_candidates)

        self._prev_gray:  Optional[np.ndarray] = None
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ─────────────────────────────────────────────────────────────────
    def find(
        self,
        frame_bgr: np.ndarray,
        exclude_player_bboxes: Optional[list[tuple[float, float, float, float]]] = None,
    ) -> list[_MotionCandidate]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Downscale for speed
        ds = self.downscale
        if ds < 1.0:
            small = cv2.resize(frame_bgr,
                               (int(frame_bgr.shape[1] * ds),
                                int(frame_bgr.shape[0] * ds)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return []

        # Frame difference
        diff = cv2.absdiff(gray, self._prev_gray)
        _, mask = cv2.threshold(diff, self.diff_thr, 255, cv2.THRESH_BINARY)
        # Tighten: dilate slightly so a fast streak forms a single blob
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)

        # Mask out player bboxes — their motion isn't ball motion.
        # Bboxes are in full-frame coords, so we scale them down to mask coords.
        if exclude_player_bboxes:
            for x1, y1, x2, y2 in exclude_player_bboxes:
                mx1 = max(0, int(x1 * ds))
                my1 = max(0, int(y1 * ds))
                mx2 = min(mask.shape[1], int(x2 * ds))
                my2 = min(mask.shape[0], int(y2 * ds))
                # Inflate by 8 px to absorb pose-keypoint slack at boundaries
                mx1 = max(0, mx1 - 8); my1 = max(0, my1 - 8)
                mx2 = min(mask.shape[1], mx2 + 8); my2 = min(mask.shape[0], my2 + 8)
                if mx2 > mx1 and my2 > my1:
                    mask[my1:my2, mx1:mx2] = 0

        # Connected components → candidate boxes
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cands: list[tuple[float, _MotionCandidate]] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # Translate area threshold back to full-frame units (area scales with ds²)
            scale = 1.0 / (ds * ds)
            full_area = area * scale
            if full_area < self.min_area or full_area > self.max_area:
                continue
            if w == 0 or h == 0:
                continue
            ar = w / h if h >= w else h / w
            ar_inv = 1.0 / ar if ar > 0 else 99
            # Aspect ratio: require not wildly elongated
            if max(w / h, h / w) > self.aspect_max:
                continue

            # Translate cell coords back to full-frame coords
            fx1 = x / ds
            fy1 = y / ds
            fx2 = (x + w) / ds
            fy2 = (y + h) / ds
            # "Brightness" of motion: stronger diff = more confident
            roi = diff[y:y + h, x:x + w]
            mean_diff = float(roi.mean()) if roi.size > 0 else 0.0
            # Map mean_diff [22..120] → conf [synth_conf .. synth_conf+0.3]
            conf = self.synth_conf + min(0.30, max(0.0, (mean_diff - 22) / 98.0) * 0.30)
            cands.append((conf, _MotionCandidate(
                x1=fx1, y1=fy1, x2=fx2, y2=fy2,
                confidence=conf, class_id=BALL_CLASS,
            )))

        # Update prev for next call
        self._prev_gray = gray

        if not cands:
            return []
        # Keep top-K by confidence
        cands.sort(key=lambda kv: -kv[0])
        return [c for _, c in cands[:self.max_cands]]
