"""
court_mapper.py — Top-down 2D court view, zone classification, ROI ball detection.

Extends CourtDetector by composing its pixel→metre homography with a
metre→top-down-pixel scaling to produce a bird-eye warp of the live frame.
Adds ROI-based ball search that prioritises the region near player hands.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

_W, _H = 40.0, 20.0   # IHF court: 40 m long × 20 m wide

# ── Zone definitions ──────────────────────────────────────────────────────────

_ZONES: list[tuple] = [
    (lambda x, y: x < 0.0,                       "goal_left"),
    (lambda x, y: x > _W,                         "goal_right"),
    (lambda x, y: x <= 6.0  and 5.5 < y < 14.5,  "area_6m_left"),
    (lambda x, y: x >= 34.0 and 5.5 < y < 14.5,  "area_6m_right"),
    (lambda x, y: 6.0 < x <= 9.0  and 2.0 < y < 18.0, "free_throw_left"),
    (lambda x, y: 31.0 <= x < 34.0 and 2.0 < y < 18.0, "free_throw_right"),
    (lambda x, y: x <= 20.0 and (y < 5.0 or y > 15.0),  "wing_left"),
    (lambda x, y: x > 20.0  and (y < 5.0 or y > 15.0),  "wing_right"),
    (lambda x, y: 9.0 < x <= 20.0,                "back_left"),
    (lambda x, y: 20.0 < x < 31.0,                "back_right"),
]


def classify_zone(x_m: float, y_m: float) -> str:
    for cond, label in _ZONES:
        if cond(x_m, y_m):
            return label
    return "centre"


@dataclass
class CourtPosition:
    x_m:   float   # metres along court length (0 = left goal line)
    y_m:   float   # metres along court width  (0 = bottom sideline)
    zone:  str     # IHF zone label
    x_td:  int     # pixel column in top-down view
    y_td:  int     # pixel row    in top-down view


# ── Court mapper ──────────────────────────────────────────────────────────────

class CourtMapper:
    """
    Wraps CourtDetector's homography to produce:
      • top-down warped frame      (top_down_view)
      • pixel → CourtPosition      (pixel_to_court)
      • annotated top-down canvas  (annotate)
      • blank court diagram        (draw_court_lines)

    Inject the pixel→metre matrix from CourtDetector._H via set_homography().
    """

    def __init__(self, output_w: int = 800, output_h: int = 400):
        self._ow = output_w
        self._oh = output_h
        self._sx = output_w / _W
        self._sy = output_h / _H
        self._H: Optional[np.ndarray] = None    # pixel → metre
        self._M: Optional[np.ndarray] = None    # pixel → top-down pixel (composed)

    def set_homography(self, H_pixel_to_metre: np.ndarray) -> None:
        self._H = H_pixel_to_metre.astype(np.float64)
        S = np.array([[self._sx, 0, 0],
                      [0, self._sy, 0],
                      [0, 0,        1]], np.float64)
        self._M = S @ self._H

    @property
    def is_ready(self) -> bool:
        return self._M is not None

    # ── Core transforms ───────────────────────────────────────────────────────

    def top_down_view(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Warp the live camera frame into a 2D bird-eye court view."""
        if not self.is_ready:
            return None
        return cv2.warpPerspective(frame, self._M, (self._ow, self._oh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(20, 35, 25))

    def pixel_to_court(self, px: float, py: float) -> Optional[CourtPosition]:
        if not self.is_ready:
            return None
        pt = cv2.perspectiveTransform(
            np.array([[[px, py]]], np.float32), self._H.astype(np.float32))
        x_m, y_m = float(pt[0, 0, 0]), float(pt[0, 0, 1])
        x_td = int(np.clip(x_m * self._sx, 0, self._ow - 1))
        y_td = int(np.clip(y_m * self._sy, 0, self._oh - 1))
        return CourtPosition(x_m=round(x_m, 2), y_m=round(y_m, 2),
                             zone=classify_zone(x_m, y_m),
                             x_td=x_td, y_td=y_td)

    # ── Annotation ────────────────────────────────────────────────────────────

    _TEAM_COLORS = {
        "team_a":    (100, 80,  255),
        "team_b":    (255, 80,  80),
        "goalkeeper":(80,  220, 80),
        "ball":      (0,   220, 255),
        "unknown":   (180, 180, 180),
    }

    def annotate(self, top_down: np.ndarray, positions: list[dict]) -> np.ndarray:
        """
        Draw players and ball on top-down view.
        Each dict: {x_m, y_m, team, label (optional), track_id (optional)}
        """
        out = top_down.copy()
        for p in positions:
            x_td = int(np.clip(p["x_m"] * self._sx, 0, self._ow - 1))
            y_td = int(np.clip(p["y_m"] * self._sy, 0, self._oh - 1))
            col  = self._TEAM_COLORS.get(p.get("team", "unknown"), (180, 180, 180))
            is_ball = p.get("team") == "ball"

            r = 5 if is_ball else 8
            cv2.circle(out, (x_td, y_td), r, col, -1, cv2.LINE_AA)
            cv2.circle(out, (x_td, y_td), r, (255, 255, 255), 1, cv2.LINE_AA)

            lbl = p.get("label") or str(p.get("track_id", ""))
            if lbl and not is_ball:
                cv2.putText(out, lbl, (x_td + r + 2, y_td + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def draw_court_lines(self) -> np.ndarray:
        """Return a blank top-down court diagram (no camera frame needed)."""
        img = np.full((self._oh, self._ow, 3), (20, 35, 25), np.uint8)

        def px(xm: float, ym: float) -> tuple[int, int]:
            return int(xm * self._sx), int(ym * self._sy)

        w, c = (220, 220, 220), (0, 180, 200)

        cv2.rectangle(img, px(0, 0), px(_W, _H), w, 2)
        cv2.line(img, px(20, 0), px(20, _H), w, 1)
        cv2.rectangle(img, px(0, 5.5), px(6, 14.5), c, 1)
        cv2.rectangle(img, px(34, 5.5), px(_W, 14.5), c, 1)
        cv2.rectangle(img, px(-1, 8.5), px(0, 11.5),  (80, 120, 255), 2)
        cv2.rectangle(img, px(_W, 8.5), px(_W+1, 11.5), (80, 120, 255), 2)
        cv2.circle(img, px(7,  10), 4, (0,  0, 200), -1)
        cv2.circle(img, px(33, 10), 4, (0,  0, 200), -1)

        dash_col = (200, 120, 0)
        for ym in np.arange(0, _H, 0.8):
            y0, y1 = int(ym * self._sy), int(min((ym + 0.45) * self._sy, self._oh))
            cv2.line(img, (int(9  * self._sx), y0), (int(9  * self._sx), y1), dash_col, 1)
            cv2.line(img, (int(31 * self._sx), y0), (int(31 * self._sx), y1), dash_col, 1)

        _zone_labels = [
            (px(3, 10),    "L6"), (px(7.5, 10), "L9"), (px(14, 3),   "LW"),
            (px(15, 10),   "LB"), (px(25, 10),  "RB"), (px(26, 17),  "RW"),
            (px(32.5, 10), "R9"), (px(37, 10),  "R6"),
        ]
        for pt, lbl in _zone_labels:
            cv2.putText(img, lbl, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (140, 140, 140), 1, cv2.LINE_AA)
        return img


# ── ROI-based ball detector ───────────────────────────────────────────────────

class ROIBallDetector:
    """
    Search for the ball near player hand/upper-body regions first.
    Falls back to full-frame detection if no ROI match is found.

    detector_fn(crop_bgr) → list of (x1, y1, x2, y2, confidence) in crop coords.
    """

    ROI_SIZE       = 120   # pixels around each hand to search
    BALL_DIAM_MIN  = 8
    BALL_DIAM_MAX  = 70

    def __init__(self, fallback_full_frame: bool = True):
        self._fallback = fallback_full_frame

    def detect(
        self,
        frame: np.ndarray,
        tracks: list,
        detector_fn,
        pose_results=None,
    ) -> Optional[tuple[float, float, float]]:
        """
        Returns (cx_px, cy_px, confidence) in full-frame coordinates, or None.
        Searches ROIs near player hands first, then falls back to full frame.
        """
        fh, fw = frame.shape[:2]
        rois   = self._build_rois(fw, fh, tracks, pose_results)

        for rx, ry, rw, rh in rois:
            crop = frame[ry:ry+rh, rx:rx+rw]
            if crop.size == 0:
                continue
            for x1, y1, x2, y2, conf in (detector_fn(crop) or []):
                diam = max(x2 - x1, y2 - y1)
                if self.BALL_DIAM_MIN <= diam <= self.BALL_DIAM_MAX:
                    return rx + (x1+x2)/2, ry + (y1+y2)/2, float(conf)

        if self._fallback:
            hits = detector_fn(frame) or []
            if hits:
                x1, y1, x2, y2, conf = max(hits, key=lambda h: h[4])
                return (x1+x2)/2, (y1+y2)/2, float(conf)

        return None

    def _build_rois(
        self,
        fw: int, fh: int,
        tracks: list,
        pose_results,
    ) -> list[tuple[int, int, int, int]]:
        s    = self.ROI_SIZE
        rois = []

        # Best source: pose wrist keypoints
        if pose_results:
            for pr in pose_results:
                kp = pr.keypoints
                for wi in (9, 10):      # L_WRIST=9, R_WRIST=10
                    if kp.conf[wi] >= 0.35:
                        wx = int(kp.xy[wi, 0])
                        wy = int(kp.xy[wi, 1])
                        rx = max(wx - s//2, 0);  ry = max(wy - s//2, 0)
                        rois.append((rx, ry,
                                     min(s, fw-rx), min(s, fh-ry)))

        # Fallback: upper portion of each bounding box
        if not rois:
            for t in tracks:
                if getattr(t, "is_ball", False):
                    continue
                bw = t.x2 - t.x1
                bh = t.y2 - t.y1
                rx = max(int(t.x1 - bw * 0.15), 0)
                ry = max(int(t.y1 + bh * 0.05), 0)
                rw = min(int(bw * 1.3), fw - rx)
                rh = min(int(bh * 0.45), fh - ry)
                if rw > 10 and rh > 10:
                    rois.append((rx, ry, rw, rh))

        return rois


# ── Pipeline integration helper ───────────────────────────────────────────────

def tracks_to_court_positions(
    tracks: list,
    mapper: CourtMapper,
) -> list[dict]:
    """
    Convert tracker output (x1/y1/x2/y2 bbox in pixels) to court positions.
    Returns list of dicts ready for CourtMapper.annotate() and API broadcast.
    """
    out = []
    for t in tracks:
        cx = (t.x1 + t.x2) / 2
        cy = (t.y1 + t.y2) / 2   # use foot point for grounded players
        foot_y = t.y2              # bottom of bounding box ≈ feet

        pos = mapper.pixel_to_court(cx, foot_y)
        if pos is None:
            continue

        team = "ball" if getattr(t, "is_ball", False) else getattr(t, "team", "unknown")
        out.append({
            "track_id": t.track_id,
            "team":     team,
            "label":    getattr(t, "label", "") or str(t.track_id),
            "x_m":      pos.x_m,
            "y_m":      pos.y_m,
            "zone":     pos.zone,
            "x_td":     pos.x_td,
            "y_td":     pos.y_td,
        })
    return out
