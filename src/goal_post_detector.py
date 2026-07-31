"""
Dynamic Geometric Goal-Post Detector.
Finds the goal based on vertical edges and geometry, ignoring colors.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

@dataclass
class GoalBox:
    x_left: float
    y_top: float
    x_right: float
    y_bottom: float
    side: str
    conf: float

    @property
    def cx(self) -> float: return (self.x_left + self.x_right) * 0.5
    @property
    def cy(self) -> float: return (self.y_top + self.y_bottom) * 0.5
    @property
    def width(self) -> float: return self.x_right - self.x_left
    @property
    def height(self) -> float: return self.y_bottom - self.y_top

    def contains_point(self, x: float, y: float, pad: float = 0.0) -> bool:
        return (self.x_left - pad <= x <= self.x_right + pad
                and self.y_top - pad <= y <= self.y_bottom + pad)

class GoalPostDetector:
    def __init__(
        self, frame_w: int, frame_h: int, run_every: int = 8, ema_alpha: float = 0.30,
        min_post_h_frac: float = 0.08, max_post_w_frac: float = 0.05,
        min_pair_dx_frac: float = 0.05, max_pair_dx_frac: float = 0.55,
    ):
        self.fw, self.fh = int(frame_w), int(frame_h)
        self.run_every = max(1, int(run_every))
        self.ema_alpha = float(ema_alpha)

        self.min_post_h_px = int(self.fh * min_post_h_frac)
        self.max_post_w_px = max(2, int(self.fw * max_post_w_frac))
        self.min_pair_dx = int(self.fw * min_pair_dx_frac)
        self.max_pair_dx = int(self.fw * max_pair_dx_frac)

        # Kernel for vertical lines extraction
        self._k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
        
        self._frame_n = 0
        self._smooth: Optional[GoalBox] = None
        self._misses = 0

    def update(self, frame: np.ndarray, frame_n: int) -> Optional[GoalBox]:
        self._frame_n = int(frame_n)
        if frame_n % self.run_every != 0 and self._smooth is not None:
            return self._smooth
        
        raw = self._detect_geometric(frame)
        
        if raw is None:
            self._misses += 1
            if self._misses > (3 * 25 // self.run_every):
                self._smooth = None
            return self._smooth

        self._misses = 0
        if self._smooth is None:
            self._smooth = raw
        else:
            a = self.ema_alpha
            self._smooth = GoalBox(
                x_left   = self._smooth.x_left   * (1 - a) + raw.x_left   * a,
                y_top    = self._smooth.y_top    * (1 - a) + raw.y_top    * a,
                x_right  = self._smooth.x_right  * (1 - a) + raw.x_right  * a,
                y_bottom = self._smooth.y_bottom * (1 - a) + raw.y_bottom * a,
                side     = raw.side, conf = self._smooth.conf * (1 - a) + raw.conf * a,
            )
        return self._smooth

    @property
    def current(self) -> Optional[GoalBox]:
        return self._smooth

    def _detect_geometric(self, frame: np.ndarray) -> Optional[GoalBox]:
        # 1. Grayscale & Edge Detection (Ignores color)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 2. Extract strictly vertical edges
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, self._k_vert)

        # 3. Find contours of these vertical lines
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < self.min_post_h_px or w > self.max_post_w_px:
                continue
            ar = h / max(w, 1)
            if ar < 3.0: 
                continue # Must be a tall rectangle
            
            # Score favors taller lines slightly higher up
            up_bonus = 1.0 if y < self.fh * 0.65 else 0.5
            score = (h / self.fh) * up_bonus
            candidates.append((x, y, w, h, score))

        if len(candidates) < 2: return None

        candidates.sort(key=lambda c: -c[4])
        candidates = candidates[:8] # Test top 8 strongest vertical lines

        # 4. Pair them geometrically
        best_pair = None
        best_score = -1.0
        
        for i in range(len(candidates)):
            xi, yi, wi, hi, si = candidates[i]
            for j in range(i + 1, len(candidates)):
                xj, yj, wj, hj, sj = candidates[j]
                
                if xj < xi:
                    xi, yi, wi, hi, si, xj, yj, wj, hj, sj = xj, yj, wj, hj, sj, xi, yi, wi, hi, si
                
                cx_l, cx_r = xi + wi//2, xj + wj//2
                dx = cx_r - cx_l
                
                if dx < self.min_pair_dx or dx > self.max_pair_dx: continue
                if abs(yi - yj) > min(hi, hj) * 0.5: continue
                if abs(hi - hj) / max(hi, hj) > 0.4: continue # Posts must be roughly same height
                
                pair_top = (yi + yj) * 0.5
                pair_bottom = pair_top + (hi + hj) * 0.5
                pair_score = si + sj
                
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (cx_l, pair_top, cx_r, pair_bottom)

        if best_pair is None: return None

        x_l, y_t, x_r, y_b = best_pair
        cx_pair = (x_l + x_r) * 0.5
        side = "left" if cx_pair < self.fw * 0.45 else ("right" if cx_pair > self.fw * 0.55 else "unknown")

        return GoalBox(float(x_l), float(y_t), float(x_r), float(y_b), side, min(1.0, best_score))

    def draw_overlay(self, img: np.ndarray) -> None:
        b = self._smooth
        if b is None: return
        x1, y1, x2, y2 = int(b.x_left), int(b.y_top), int(b.x_right), int(b.y_bottom)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"GOAL[{b.side}]", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)