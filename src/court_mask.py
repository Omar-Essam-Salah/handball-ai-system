"""
Court Region Mask V2 — colour-agnostic.

Why V1 broke
------------
Version 1 hue-locked on the first 5 warmup frames. If the video opened on an
intro shot / a wide angle dominated by crowd, the locked hue was wrong forever.
A different match (Rio 2016 = green+red, Sweden = blue+yellow) couldn't be
processed with the same locked hue. Result: visibility stayed near 0.02,
graphic-frame guard fired on every frame, all tracking was paused.

V2 design
---------
The court is the *largest, well-saturated, well-lit, mostly-bottom-of-frame
connected region*. We don't need to know its colour at all.

Algorithm (every infer frame):
    1. Downscale frame to 320 wide.
    2. Convert to HSV.
    3. Build "candidate court" mask: saturation ≥ S_MIN AND V_MIN ≤ value ≤ V_MAX.
       (kicks out shadows, specular highlights, near-greyscale crowd patches)
    4. Morphological close (15×15) to fuse painted markings + line gaps.
    5. Largest connected component → tentative court.
    6. Plausibility checks:
        - blob area ≥ MIN_FRAC of frame, AND
        - blob's vertical centroid is in the lower 65% of the frame.
       Either fails → no court visible (graphic / behind-net / replay).
    7. Optional refinement: once a stable court is found for ≥ N frames, lock
       the dominant HSV centre of the blob and bias subsequent detections
       toward that hue. If conditions change, the lock auto-resets.

Cost: ~2-4 ms / call at 320×180. Cheap enough to run every infer frame.
"""

from __future__ import annotations
from collections import deque
from typing import Optional

import cv2
import numpy as np


# ── Tunables ────────────────────────────────────────────────────────────────
_S_MIN          = 35     # saturation floor — anything below is greyscale crowd
_V_MIN          = 50     # value  floor — kicks out shadows
_V_MAX          = 235    # value  ceiling — kicks out blown-out highlights
_MIN_FRAC       = 0.06   # minimum visibility to count as "play frame"
# Reject blobs whose centroid sits in the top portion of the frame (= crowd).
# Court centroids typically land at 0.45-0.65 of the height, so anything with
# centroid Y above this fraction is a crowd / banner / sponsor blob.
_TOP_REJECT_Y   = 0.30
_LOCK_FRAMES    = 18     # how many consecutive good frames before we lock hue
_LOCK_HUE_TOL   = 22     # ± degrees once locked — biases mask toward locked hue
_RECAL_BAD_FN   = 30     # recalibrate (drop lock) after this many bad frames
_FOOT_PAD_PX    = 6      # absorb feet that just kiss the touchline


class CourtMask:
    def __init__(
        self,
        frame_w:  int,
        frame_h:  int,
        run_every: int = 4,
    ):
        self.fw, self.fh = int(frame_w), int(frame_h)
        self.run_every  = max(1, int(run_every))

        self._small_w   = 320
        self._small_h   = max(64, int(self._small_w * frame_h / max(frame_w, 1)))
        self._k_close   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._k_open    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._frame_n   = 0
        self._mask_small: Optional[np.ndarray] = None
        self._mask:       Optional[np.ndarray] = None
        self._visibility: float = 0.0

        # Optional hue lock — biases detection once we've seen a stable court
        self._locked_hue:  Optional[int] = None
        self._good_streak: int  = 0
        self._bad_streak:  int  = 0

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, frame_n: int) -> None:
        self._frame_n = int(frame_n)
        if frame is None or frame.size == 0:
            return
        if self._mask is not None and (frame_n % self.run_every) != 0:
            return
        self._compute(frame)

    def is_in_court(self, x: float, y: float) -> bool:
        if self._mask_small is None:
            # Fail-open: before first compute, accept everything so we don't kill tracking
            return True
        sw, sh = self._small_w, self._small_h
        sx = int(x * sw / max(self.fw, 1))
        sy = int(y * sh / max(self.fh, 1))
        if sx < 0 or sy < 0 or sx >= sw or sy >= sh:
            return False
        # 1-cell pad so feet on the touchline still count
        x0, y0 = max(0, sx - 1), max(0, sy - 1)
        x1, y1 = min(sw, sx + 2), min(sh, sy + 2)
        return bool(self._mask_small[y0:y1, x0:x1].any())

    def is_play_frame(self) -> bool:
        return self._visibility >= _MIN_FRAC

    @property
    def visibility(self) -> float:
        return self._visibility

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def stats(self) -> dict:
        return {
            "visibility":  round(self._visibility, 3),
            "is_play":     self.is_play_frame(),
            "locked_hue":  self._locked_hue,
            "good_streak": self._good_streak,
            "bad_streak":  self._bad_streak,
        }

    def draw_overlay(self, img: np.ndarray, alpha: float = 0.30) -> None:
        if self._mask is None:
            return
        m = self._mask
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        out_of_court = (m == 0)
        if out_of_court.any():
            img[out_of_court] = (img[out_of_court] * (1.0 - alpha)).astype(np.uint8)
        col = (200, 230, 255) if self.is_play_frame() else (60, 60, 220)
        cv2.putText(img, f"COURT {self._visibility*100:.0f}%",
                    (12, img.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────────────
    def _compute(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, (self._small_w, self._small_h),
                            interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_arr = hsv[..., 0]
        s_arr = hsv[..., 1]
        v_arr = hsv[..., 2]

        # 1. Discover the dominant hue inside the lower 70 % of the frame.
        # This is what the "court colour" looks like THIS frame — it adapts to
        # different matches (Sweden blue, Rio green, etc.) and to camera changes.
        dominant_hue = self._find_dominant_hue(h_arr, s_arr, v_arr)
        if dominant_hue is None:
            self._on_no_court()
            return

        # 2. Build a tight hue+S/V mask centred on that hue.
        tol = _LOCK_HUE_TOL
        lo, hi = (dominant_hue - tol) % 180, (dominant_hue + tol) % 180
        if lo <= hi:
            hue_band = ((h_arr >= lo) & (h_arr <= hi))
        else:
            hue_band = ((h_arr >= lo) | (h_arr <= hi))
        cand = hue_band & (s_arr >= _S_MIN) & (v_arr >= _V_MIN) & (v_arr <= _V_MAX)

        # 3. Optional: also accept pixels that match the previously-locked hue —
        #    helps two-tone courts (blue + yellow accent) hold together.
        if self._locked_hue is not None and self._locked_hue != dominant_hue:
            lo2, hi2 = (self._locked_hue - tol) % 180, (self._locked_hue + tol) % 180
            if lo2 <= hi2:
                band2 = ((h_arr >= lo2) & (h_arr <= hi2))
            else:
                band2 = ((h_arr >= lo2) | (h_arr <= hi2))
            cand = cand | (band2 & (s_arr >= _S_MIN)
                                 & (v_arr >= _V_MIN) & (v_arr <= _V_MAX))

        mask = (cand.astype(np.uint8) * 255)

        # 4. Morphology — fuse line markings + drop pepper noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)

        # 5. Largest connected component
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            self._on_no_court()
            return

        # Skip background label 0 → pick the biggest blob
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest_idx = 1 + int(np.argmax(areas))
        biggest_area = int(stats[biggest_idx, cv2.CC_STAT_AREA])
        total_pixels = self._small_w * self._small_h
        visibility = biggest_area / max(total_pixels, 1)

        if visibility < _MIN_FRAC * 0.6:    # nowhere near plausible
            self._on_no_court()
            return

        # 5. Plausibility: blob centroid must NOT be in the top portion (crowd).
        cy_blob = stats[biggest_idx, cv2.CC_STAT_TOP] \
                  + stats[biggest_idx, cv2.CC_STAT_HEIGHT] * 0.5
        if cy_blob < self._small_h * _TOP_REJECT_Y:
            # centroid is in the upper third — crowd / banner / sponsor; not the court
            self._on_no_court()
            return

        court_only = (labels == biggest_idx).astype(np.uint8) * 255
        # Dilate slightly so feet on the touchline still register as in-court
        if _FOOT_PAD_PX > 0:
            kp = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_FOOT_PAD_PX * 2 + 1, _FOOT_PAD_PX * 2 + 1))
            court_only = cv2.dilate(court_only, kp, iterations=1)

        self._mask_small = court_only
        self._mask = cv2.resize(court_only, (self.fw, self.fh),
                                 interpolation=cv2.INTER_NEAREST)
        self._visibility = float(visibility)
        self._on_court(hsv, court_only)

    def _find_dominant_hue(self, h_arr: np.ndarray, s_arr: np.ndarray,
                              v_arr: np.ndarray) -> Optional[int]:
        """Histogram-based dominant hue from the lower 70 % of the frame
        (skips top crowd band). Re-evaluated every call — no permanent lock.
        Returns ``None`` if not enough saturated pixels to decide."""
        top = int(self._small_h * 0.30)
        sub_h = h_arr[top:]
        sub_s = s_arr[top:]
        sub_v = v_arr[top:]
        good = (sub_s >= 50) & (sub_v >= 60) & (sub_v <= 230)
        if good.sum() < 400:
            return None
        hues = sub_h[good].astype(np.int32)
        hist = np.bincount(hues, minlength=180).astype(np.float32)
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 11), 0).flatten()
        peak = int(np.argmax(hist))
        # Sanity: peak strength must exceed mean by 3× to count as "dominant"
        if hist[peak] < hist.mean() * 3.0:
            return None
        return peak

    def _on_no_court(self) -> None:
        self._mask = None
        self._mask_small = None
        self._visibility = 0.0
        self._good_streak = 0
        self._bad_streak += 1
        # Drop the hue lock if conditions have been bad for a while —
        # this lets the next match / camera angle re-discover the court colour.
        if self._bad_streak >= _RECAL_BAD_FN:
            self._locked_hue = None

    def _on_court(self, hsv_small: np.ndarray, court_mask_small: np.ndarray) -> None:
        self._bad_streak = 0
        self._good_streak += 1

        # Lock the dominant hue inside the court blob once we've been stable
        if self._good_streak >= _LOCK_FRAMES and self._locked_hue is None:
            inside = court_mask_small > 0
            if inside.any():
                hues = hsv_small[..., 0][inside]
                sats = hsv_small[..., 1][inside]
                # Only consider well-saturated pixels for hue voting (kicks out painted lines)
                good = sats >= 80
                if good.sum() >= 200:
                    hist = np.bincount(hues[good].astype(np.int32),
                                       minlength=180).astype(np.float32)
                    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 9), 0).flatten()
                    self._locked_hue = int(np.argmax(hist))
