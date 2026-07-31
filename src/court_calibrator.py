"""
court_calibrator.py — Fully-automatic handball court calibration.

Pipeline
────────
1. HSV + adaptive white-line mask
2. LSD (LineSegmentDetector) — sub-pixel accurate segment extraction
3. Angle-based clustering → horizontal / vertical / curve candidates
4. RANSAC line fitting → two sidelines + up to two baselines
5. Coarse 4-corner DLT → rough pixel→metre homography
6. Metric projection of all LSD segments (using rough H)
7. Feature search in metric space:
     • x = 6 / x = 34 vertical straight sections of the 6m area
     • x = 9 / x = 31 straight sections of the 9m free-throw line
     • x = 20 centre line
     • 6m arcs — RANSAC Taubin circle fit on curved segment endpoints
8. Multi-point RANSAC homography from all pixel↔metric correspondences
9. Levenberg–Marquardt reprojection refinement (if scipy available)
10. Geometric validation (aspect ratio, reprojection error)

Output
──────
Fills CourtDetector._H / ._iH in-place via set_homography_direct().
All coordinates use IHF convention:
    x ∈ [0, 40]  along court length (left goal = 0, right goal = 40)
    y ∈ [0, 20]  along court width
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── IHF constants ─────────────────────────────────────────────────────────────

W, H     = 40.0, 20.0
R6, R9   = 6.0,  9.0
_GP_LEFT  = [(0.0,  8.5), (0.0, 11.5)]  # goal-post centres, left goal
_GP_RIGHT = [(40.0, 8.5), (40.0, 11.5)]

# Metric x-positions of known vertical line features
_KNOWN_VERTICALS = {6.0: "6m_left", 34.0: "6m_right",
                    9.0: "9m_left", 31.0: "9m_right",
                    20.0: "centre"}

# HSV thresholds for white line detection
_WHITE_LO  = np.array([  0,   0, 160], np.uint8)
_WHITE_HI  = np.array([180,  55, 255], np.uint8)

_MIN_SEG_LEN  = 25   # px — discard shorter segments
_VERT_TOL_M   = 0.5  # metres — band around known x-positions to accept segments
_ARC_TOL_PX   = 8    # pixels — RANSAC inlier tolerance for circle fit
_REPROJ_LIMIT = 8.0  # pixels — validation threshold
_MAX_CALIB_S  = 3.0  # seconds — hard timeout for calibrate()


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Segment:
    x1: float; y1: float; x2: float; y2: float
    length: float = 0.0
    angle: float  = 0.0   # degrees, -90 to 90

    def __post_init__(self):
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.length = math.hypot(dx, dy)
        self.angle  = math.degrees(math.atan2(dy, dx))

    @property
    def midpoint(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def endpoints(self) -> np.ndarray:
        return np.array([[self.x1, self.y1], [self.x2, self.y2]], np.float32)


@dataclass
class CalibResult:
    success:         bool
    H:               Optional[np.ndarray]     # pixel → metre (3×3 float32)
    reproj_error_px: float   = 999.0
    n_inliers:       int     = 0
    features_found:  list    = field(default_factory=list)
    elapsed_ms:      float   = 0.0
    debug_frame:     Optional[np.ndarray] = None


# ── Core math helpers ─────────────────────────────────────────────────────────

def _intersect_hough(r1, t1, r2, t2) -> Optional[tuple[float, float]]:
    """Analytical intersection of two Hough lines (ρ, θ)."""
    A = np.array([[math.cos(t1), math.sin(t1)],
                  [math.cos(t2), math.sin(t2)]], np.float64)
    b = np.array([r1, r2], np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    x, y = np.linalg.solve(A, b)
    return float(x), float(y)


def _circle_from_3pts(p1, p2, p3) -> Optional[tuple[float, float, float]]:
    """Circumscribed circle of 3 points. Returns (cx, cy, r) or None."""
    ax, ay = p1; bx, by = p2; cx_, cy_ = p3
    D = 2.0 * (ax*(by - cy_) + bx*(cy_ - ay) + cx_*(ay - by))
    if abs(D) < 1e-10:
        return None
    sq = lambda x, y: x*x + y*y
    ux = (sq(ax,ay)*(by-cy_) + sq(bx,by)*(cy_-ay) + sq(cx_,cy_)*(ay-by)) / D
    uy = (sq(ax,ay)*(cx_-bx) + sq(bx,by)*(ax-cx_) + sq(cx_,cy_)*(bx-ax)) / D
    return ux, uy, math.hypot(ax-ux, ay-uy)


def _ransac_circle(
    pts: np.ndarray,
    tolerance: float = _ARC_TOL_PX,
    iters: int = 400,
    min_inliers: int = 12,
) -> Optional[tuple[float, float, float]]:
    """
    RANSAC circle fit on a (N,2) point cloud.
    Returns (cx, cy, r) with the most inliers, or None.
    """
    if len(pts) < min_inliers:
        return None
    best, best_n = None, 0
    rng = np.random.default_rng(42)
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        result = _circle_from_3pts(*pts[idx])
        if result is None:
            continue
        cx, cy, r = result
        dists = np.abs(np.hypot(pts[:,0]-cx, pts[:,1]-cy) - r)
        n = int((dists < tolerance).sum())
        if n > best_n:
            best_n = n
            best   = (cx, cy, r)

    if best is None or best_n < min_inliers:
        return None

    # Refine on inliers using least-squares
    cx, cy, r = best
    dists = np.abs(np.hypot(pts[:,0]-cx, pts[:,1]-cy) - r)
    inliers = pts[dists < tolerance]
    # Taubin algebraic refinement on inliers
    refined = _taubin_circle(inliers)
    return refined if refined else best


def _taubin_circle(pts: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Taubin algebraic circle fit. Numerically stable, no initial estimate."""
    if len(pts) < 3:
        return None
    X = pts[:, 0].astype(np.float64)
    Y = pts[:, 1].astype(np.float64)
    mx, my = X.mean(), Y.mean()
    Xc, Yc = X - mx, Y - my
    Z = Xc**2 + Yc**2
    Zm = Z.mean()
    if Zm < 1e-10:
        return None
    Z0 = (Z - Zm) / (2.0 * math.sqrt(Zm))
    M = np.column_stack([Z0, Xc, Yc, np.ones(len(pts))])
    try:
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        A = Vt[-1]
    except np.linalg.LinAlgError:
        return None
    if abs(A[0]) < 1e-12:
        return None
    A0 = A[0] / (2.0 * math.sqrt(Zm))
    cx = -A[1] / (2.0 * A0) + mx
    cy = -A[2] / (2.0 * A0) + my
    disc = A[1]**2 + A[2]**2 - 4.0 * A0 * A[3]
    if disc < 0:
        return None
    r = math.sqrt(disc) / (2.0 * abs(A0))
    return cx, cy, r


def _reproj_error(H: np.ndarray, px_pts: np.ndarray, m_pts: np.ndarray) -> float:
    """Mean reprojection error: transform px_pts by H, compare with m_pts."""
    if len(px_pts) == 0:
        return 999.0
    proj = cv2.perspectiveTransform(px_pts.reshape(-1,1,2).astype(np.float32), H)
    return float(np.mean(np.hypot(proj[:,0,0] - m_pts[:,0],
                                   proj[:,0,1] - m_pts[:,1])))


def _sort_quad(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points as [TL, TR, BR, BL]."""
    pts = pts[np.argsort(pts[:,1])]
    top = pts[:2][np.argsort(pts[:2,0])]
    bot = pts[2:][np.argsort(pts[2:,0])]
    return np.array([top[0], top[1], bot[1], bot[0]], np.float32)


# ── LSD wrapper ───────────────────────────────────────────────────────────────

def _run_lsd(gray: np.ndarray) -> list[Segment]:
    """Run LSD on a grayscale image and return filtered Segment list."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 0.8, 0.6)
    raw, *_ = lsd.detect(gray)
    if raw is None:
        return []
    segs = []
    for line in raw.reshape(-1, 4):
        s = Segment(*line)
        if s.length >= _MIN_SEG_LEN:
            segs.append(s)
    return segs


# ── White mask ────────────────────────────────────────────────────────────────

def _white_mask(frame: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _WHITE_LO, _WHITE_HI)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, -12)
    mask  = cv2.bitwise_or(mask, adapt)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              np.ones((3,3), np.uint8), iterations=2)
    return mask


# ── Boundary line detection ───────────────────────────────────────────────────

def _hough_dominant_lines(
    edges: np.ndarray,
    angle_center: float,
    angle_tol: float = 18.0,
    threshold: int = 60,
) -> list[tuple[float, float]]:
    """
    Run HoughLines and return lines whose angle is within angle_tol of angle_center.
    angle_center in degrees; returns [(rho, theta), ...] sorted by |rho|.
    """
    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold)
    if lines is None:
        return []
    result = []
    for rho, theta in lines[:, 0]:
        ang = math.degrees(theta)
        diff = abs(ang - (90.0 + angle_center)) % 180.0
        diff = min(diff, 180.0 - diff)
        if diff < angle_tol:
            result.append((rho, theta))
    result.sort(key=lambda l: l[0])
    return result


def _pick_outer_pair(lines: list[tuple[float, float]]) -> tuple | None:
    """Return the two lines with the most extreme rho values (outermost pair)."""
    if len(lines) < 2:
        return None
    return lines[0], lines[-1]


# ── Coarse 4-corner estimation ────────────────────────────────────────────────

def _coarse_corners_from_hough(
    frame: np.ndarray,
    mask: np.ndarray,
) -> Optional[list[tuple[float, float]]]:
    """
    Detect 4 court corners from Hough lines on the white mask.
    Returns [TL, TR, BR, BL] in pixel coordinates, or None.
    """
    edges = cv2.Canny(mask, 40, 120)
    fh, fw = frame.shape[:2]

    h_lines = _hough_dominant_lines(edges, angle_center=0.0)    # horizontal
    v_lines = _hough_dominant_lines(edges, angle_center=90.0)   # vertical

    pair_h = _pick_outer_pair(h_lines)
    pair_v = _pick_outer_pair(v_lines)
    if pair_h is None or pair_v is None:
        return None

    top,    bot   = pair_h
    left,   right = pair_v

    corners = [
        _intersect_hough(*top,  *left),
        _intersect_hough(*top,  *right),
        _intersect_hough(*bot,  *right),
        _intersect_hough(*bot,  *left),
    ]
    if None in corners:
        return None

    # Sanity: corners inside frame ± 15 %
    margin_x, margin_y = fw * 0.15, fh * 0.15
    for cx, cy in corners:
        if not (-margin_x < cx < fw + margin_x and -margin_y < cy < fh + margin_y):
            return None

    # Aspect-ratio sanity: court is 40×20, ratio ≥ 1.4 after de-projection
    tl, tr, br, bl = [np.array(c) for c in corners]
    top_w  = np.linalg.norm(tr - tl)
    bot_w  = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    mean_w = (top_w + bot_w) / 2
    if mean_w < left_h * 1.2:          # too square → likely wrong
        return None

    return corners


def _H_from_corners(corners: list[tuple[float, float]]) -> Optional[np.ndarray]:
    """Compute pixel→metre H from 4 corners [TL, TR, BR, BL]."""
    src = np.array(corners, np.float32)
    dst = np.array([(0,0),(W,0),(W,H),(0,H)], np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    return H


# ── Metric-space segment projection ──────────────────────────────────────────

def _project_segments(segs: list[Segment], H: np.ndarray) -> np.ndarray:
    """
    Project all segment endpoints into metric space using H.
    Returns (N, 4) array: [x1m, y1m, x2m, y2m].
    """
    pts = np.array([[s.x1, s.y1, s.x2, s.y2] for s in segs], np.float32)
    ep1 = cv2.perspectiveTransform(pts[:, :2].reshape(-1,1,2), H).reshape(-1,2)
    ep2 = cv2.perspectiveTransform(pts[:, 2:].reshape(-1,1,2), H).reshape(-1,2)
    return np.hstack([ep1, ep2])        # (N, 4)


# ── Known-vertical feature detection ─────────────────────────────────────────

def _detect_known_verticals(
    segs: list[Segment],
    metric_segs: np.ndarray,
    frame_h: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Find LSD segments that lie on known-x vertical lines (6m, 9m, centre).
    Returns list of (pixel_pt, metric_pt) correspondence pairs.
    """
    corr = []
    for i, (ms, seg) in enumerate(zip(metric_segs, segs)):
        x1m, y1m, x2m, y2m = ms
        # Both endpoints should be near the same known x
        xm_mid = (x1m + x2m) / 2
        ym_mid = (y1m + y2m) / 2
        for x_known in _KNOWN_VERTICALS:
            if abs(xm_mid - x_known) < _VERT_TOL_M:
                # Check it's roughly vertical in pixel space
                a = abs(seg.angle)
                if 60 < a < 120 or a > 170 or a < -170:
                    # Use midpoint as correspondence
                    px = np.array([seg.midpoint[0], seg.midpoint[1]], np.float32)
                    # Metric: clamp ym to valid court range
                    ym_clamped = float(np.clip(ym_mid, 0.0, H))
                    mx = np.array([x_known, ym_clamped], np.float32)
                    corr.append((px, mx))
                break
    return corr


# ── 6m arc detection ──────────────────────────────────────────────────────────

def _collect_arc_candidates(
    segs: list[Segment],
    metric_segs: np.ndarray,
    goal_side: str,    # "left" | "right"
    rough_H: np.ndarray,
    frame_shape: tuple,
) -> np.ndarray:
    """
    Collect pixel endpoints of LSD segments that fall in the expected
    region of the 6m arc (metric x ∈ [0,7] for left, [33,40] for right).
    """
    if goal_side == "left":
        x_lo, x_hi = -1.0, 7.5
        y_lo, y_hi = 1.0, 19.0
    else:
        x_lo, x_hi = 32.5, 41.0
        y_lo, y_hi = 1.0, 19.0

    pts = []
    for ms, seg in zip(metric_segs, segs):
        x1m, y1m, x2m, y2m = ms
        xm = (x1m + x2m) / 2
        ym = (y1m + y2m) / 2
        if x_lo < xm < x_hi and y_lo < ym < y_hi:
            # Exclude near-vertical segments (baselines) and long horizontals (sidelines)
            a = abs(seg.angle % 90)
            if 10 < a < 80:          # genuinely curved / diagonal → arc candidate
                pts.extend([[seg.x1, seg.y1], [seg.x2, seg.y2]])
    return np.array(pts, np.float32) if pts else np.empty((0,2), np.float32)


def _arc_to_correspondences(
    arc_result: tuple[float, float, float],
    goal_side: str,
    rough_H: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Convert a detected arc (cx_px, cy_px, r_px) into pixel↔metric correspondences.

    Strategy: sample points on the detected pixel arc and project them to the
    theoretical IHF arc using the rough homography → per-point correction.
    Generates 6 correspondence pairs per arc.
    """
    cx_px, cy_px, r_px = arc_result
    corr = []

    # Sample angles covering the visible part of the arc
    if goal_side == "left":
        gp_metric = [(0.0, 8.5), (0.0, 11.5)]
        angles_deg = np.linspace(-75, 75, 6)
    else:
        gp_metric = [(40.0, 8.5), (40.0, 11.5)]
        angles_deg = np.linspace(105, 255, 6)

    # Use the lower goal post (post at y=8.5) for arc of the lower half
    # and upper (y=11.5) for the upper half
    for ang_d in angles_deg:
        ang_r = math.radians(ang_d)
        px_pt = np.array([cx_px + r_px * math.cos(ang_r),
                           cy_px + r_px * math.sin(ang_r)], np.float32)

        # Project this pixel point to metric using rough H
        m_rough = cv2.perspectiveTransform(
            px_pt.reshape(1,1,2), rough_H)[0,0]
        xr, yr = float(m_rough[0]), float(m_rough[1])

        # Snap to nearest 6m arc point: find nearest post, compute ideal angle
        best_snap = None
        best_d    = 1e9
        for gpx, gpy in gp_metric:
            # Ideal arc point: keep same angle from post, force r=6.0
            dx, dy = xr - gpx, yr - gpy
            ideal_ang = math.atan2(dy, dx)
            snap = np.array([gpx + R6*math.cos(ideal_ang),
                              gpy + R6*math.sin(ideal_ang)], np.float32)
            # Validate inside court
            if 0 <= snap[0] <= W and 0 <= snap[1] <= H:
                d = math.hypot(xr-snap[0], yr-snap[1])
                if d < best_d:
                    best_d    = d
                    best_snap = snap

        if best_snap is not None and best_d < 2.5:   # within 2.5m of expected arc
            corr.append((px_pt, best_snap))

    return corr


# ── Sideline correspondences ──────────────────────────────────────────────────

def _sideline_correspondences(
    h_lines: list[tuple[float, float]],
    frame_shape: tuple,
    which: str,   # "top" | "bottom"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Sample points along a detected sideline (Hough rho/theta) and map them to
    y=0 (top) or y=20 (bottom) in metric space.
    x positions sampled at [2, 10, 20, 30, 38] metres.
    """
    if not h_lines:
        return []
    fh, fw = frame_shape[:2]
    rho, theta = h_lines[0]
    y_metric = 0.0 if which == "top" else H

    x_samples_m = [2.0, 10.0, 20.0, 30.0, 38.0]
    corr = []
    for xm in x_samples_m:
        # Approximate pixel x assuming no perspective (rough)
        # We'll just parameterise along the line
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Point on line closest to x=xm*(fw/W)
        # Line: x*cos_t + y*sin_t = rho
        # Use two frame-crossing points
        if abs(sin_t) > 0.1:
            px_x = xm * fw / W
            px_y = (rho - px_x * cos_t) / sin_t
        else:
            px_y = fw / 2
            px_x = (rho - px_y * sin_t) / cos_t
        if -fw*0.1 < px_x < fw*1.1 and -fh*0.1 < px_y < fh*1.1:
            corr.append((np.array([px_x, px_y], np.float32),
                         np.array([xm, y_metric], np.float32)))
    return corr


# ── Baseline correspondences ──────────────────────────────────────────────────

def _baseline_correspondences(
    v_lines: list[tuple[float, float]],
    frame_shape: tuple,
    which: str,   # "left" | "right"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample points along a detected baseline, map to x=0 or x=40."""
    if not v_lines:
        return []
    fh, fw = frame_shape[:2]
    rho, theta = v_lines[0]
    x_metric = 0.0 if which == "left" else W

    y_samples_m = [2.0, 8.5, 10.0, 11.5, 18.0]
    corr = []
    for ym in y_samples_m:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        if abs(cos_t) > 0.1:
            px_y = ym * fh / H
            px_x = (rho - px_y * sin_t) / cos_t
        else:
            px_x = fh / 2
            px_y = (rho - px_x * cos_t) / sin_t
        if -fw*0.1 < px_x < fw*1.1 and -fh*0.1 < px_y < fh*1.1:
            corr.append((np.array([px_x, px_y], np.float32),
                         np.array([x_metric, ym], np.float32)))
    return corr


# ── LM refinement (optional scipy) ───────────────────────────────────────────

def _lm_refine(
    H: np.ndarray,
    px_pts: np.ndarray,
    m_pts: np.ndarray,
) -> np.ndarray:
    """
    Levenberg–Marquardt refinement of H using reprojection error.
    Falls back silently if scipy is not available.
    """
    try:
        from scipy.optimize import least_squares

        def residuals(h8):
            H9 = np.append(h8, 1.0).reshape(3,3)
            proj = cv2.perspectiveTransform(
                px_pts.reshape(-1,1,2).astype(np.float32), H9.astype(np.float32))
            diff = proj.reshape(-1,2) - m_pts
            return diff.ravel()

        x0 = H.ravel()[:8]
        result = least_squares(residuals, x0, method='lm', max_nfev=400)
        H_ref = np.append(result.x, 1.0).reshape(3,3).astype(np.float32)
        return H_ref
    except Exception:
        return H


# ── Debug visualisation ───────────────────────────────────────────────────────

def _draw_debug(
    frame: np.ndarray,
    segs: list[Segment],
    h_lines: list,
    v_lines: list,
    arc_results: dict,
    correspondences: list,
    H: Optional[np.ndarray],
) -> np.ndarray:
    dbg = frame.copy()

    # LSD segments (dim grey)
    for s in segs:
        cv2.line(dbg, (int(s.x1),int(s.y1)), (int(s.x2),int(s.y2)),
                 (80,80,80), 1, cv2.LINE_AA)

    # Hough boundary lines (blue / green)
    fh, fw = frame.shape[:2]
    for col, lines in [((255,80,80), h_lines), ((80,255,80), v_lines)]:
        for rho, theta in lines[:2]:
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            x0, y0 = cos_t*rho, sin_t*rho
            scale  = max(fw, fh)
            pt1 = (int(x0 + scale*(-sin_t)), int(y0 + scale*cos_t))
            pt2 = (int(x0 - scale*(-sin_t)), int(y0 - scale*cos_t))
            cv2.line(dbg, pt1, pt2, col, 2, cv2.LINE_AA)

    # Detected arcs (yellow circles)
    for side, (cx,cy,r) in arc_results.items():
        cv2.circle(dbg, (int(cx), int(cy)), int(r), (0,220,255), 2, cv2.LINE_AA)
        cv2.circle(dbg, (int(cx), int(cy)), 4, (0,220,255), -1)
        cv2.putText(dbg, f"6m {side}", (int(cx)+5, int(cy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,255), 1)

    # Correspondences (magenta dots)
    for px_pt, m_pt in correspondences:
        cv2.circle(dbg, (int(px_pt[0]), int(px_pt[1])), 4, (255,0,220), -1)

    # Court overlay if H is available
    if H is not None:
        iH, _ = cv2.findHomography(
            np.array([(0,0),(W,0),(W,H),(0,H)], np.float32),
            np.array([(0,0),(W,0),(W,H),(0,H)], np.float32))
        iH, _ = cv2.findHomography(
            np.array([(0,0),(W,0),(W,H),(0,H)], np.float32), np.zeros((4,2),np.float32))
        # Just draw boundary in orange
        corners_m = np.array([(0,0),(W,0),(W,H),(0,H),(0,0)], np.float32)
        iH_mat, _ = cv2.findHomography(
            np.array([(0,0),(W,0),(W,H),(0,H)], np.float32),
            np.array([(0,0),(1,0),(1,1),(0,1)], np.float32))  # dummy
        # Use inverse of H instead
        Hi = np.linalg.inv(H)
        px_boundary = cv2.perspectiveTransform(
            corners_m.reshape(-1,1,2), Hi.astype(np.float32))
        pts_i = px_boundary.reshape(-1,2).astype(np.int32)
        for i in range(len(pts_i)-1):
            cv2.line(dbg, tuple(pts_i[i]), tuple(pts_i[i+1]), (0,140,255), 2)

    cv2.putText(dbg, "AUTO CALIBRATION DEBUG", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return dbg


# ── Main calibrator ───────────────────────────────────────────────────────────

class CourtCalibrator:
    """
    Fully-automatic handball court calibration from a single frame.

    Usage
    ─────
        calibrator = CourtCalibrator()
        result = calibrator.calibrate(frame)
        if result.success:
            detector.set_homography_direct(result.H)

    For RTSP sources:
        result = calibrator.calibrate_from_rtsp(rtsp_url, n_frames=5)
    """

    def __init__(
        self,
        min_reproj_px: float = _REPROJ_LIMIT,
        debug: bool = False,
    ):
        self.min_reproj_px = min_reproj_px
        self.debug         = debug
        self._last_result: Optional[CalibResult] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> CalibResult:
        """
        Run full calibration pipeline on a single BGR frame.
        Returns CalibResult; result.H is pixel→metre if success.
        """
        t0 = time.monotonic()
        result = self._run_pipeline(frame)
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        self._last_result  = result
        return result

    def calibrate_from_rtsp(
        self,
        rtsp_url: str,
        n_frames: int = 5,
        frame_skip: int = 15,
        auth: Optional[tuple[str, str]] = None,
    ) -> CalibResult:
        """
        Open an RTSP stream, grab n_frames separated by frame_skip frames,
        and return the best calibration result across all frames.
        """
        if auth:
            # Embed credentials into URL
            from urllib.parse import urlparse, urlunparse
            p = urlparse(rtsp_url)
            netloc = f"{auth[0]}:{auth[1]}@{p.hostname}"
            if p.port:
                netloc += f":{p.port}"
            rtsp_url = urlunparse(p._replace(netloc=netloc))

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return CalibResult(success=False, H=None,
                               features_found=["rtsp_open_failed"])

        best: Optional[CalibResult] = None
        grabbed = 0
        frame_n = 0

        while grabbed < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1
            if frame_n % frame_skip != 0:
                continue
            result = self.calibrate(frame)
            grabbed += 1
            if result.success:
                if best is None or result.reproj_error_px < best.reproj_error_px:
                    best = result

        cap.release()

        if best is None:
            return CalibResult(success=False, H=None,
                               features_found=["no_frame_succeeded"])
        return best

    @property
    def last_result(self) -> Optional[CalibResult]:
        return self._last_result

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run_pipeline(self, frame: np.ndarray) -> CalibResult:
        fh, fw = frame.shape[:2]
        features: list[str] = []

        # ── Step 1: white mask + LSD ──────────────────────────────────────────
        mask = _white_mask(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        segs = _run_lsd(masked_gray)

        if len(segs) < 10:
            return CalibResult(False, None, features_found=["too_few_segments"])

        # ── Step 2: Hough boundary lines ─────────────────────────────────────
        edges  = cv2.Canny(mask, 40, 120)
        h_lines = _hough_dominant_lines(edges, angle_center=0.0,   threshold=55)
        v_lines = _hough_dominant_lines(edges, angle_center=90.0,  threshold=55)

        if len(h_lines) < 2:
            return CalibResult(False, None, features_found=["no_sidelines"])
        features.append("sidelines")

        # ── Step 3: Coarse homography from 4 corners ─────────────────────────
        corners = _coarse_corners_from_hough(frame, mask)
        rough_H: Optional[np.ndarray] = None
        if corners:
            rough_H = _H_from_corners(corners)
            features.append("coarse_corners")

        # Fallback: if only sidelines (no full boundary), skip to partial fit
        if rough_H is None:
            # Use only the two sidelines to seed a partial H
            # This handles cameras that see only one half of the court
            rough_H = self._partial_H_from_sidelines(h_lines, v_lines, fw, fh)
            if rough_H is None:
                return CalibResult(False, None, features_found=features + ["no_rough_H"])
            features.append("partial_H_from_sidelines")

        # ── Step 4: Project segments to metric space ──────────────────────────
        metric_segs = _project_segments(segs, rough_H)

        # Filter: keep segments whose metric midpoint is within court bounds (+margin)
        mx_mid = (metric_segs[:,0] + metric_segs[:,2]) / 2
        my_mid = (metric_segs[:,1] + metric_segs[:,3]) / 2
        valid  = ((-2 < mx_mid) & (mx_mid < W+2) &
                  (-2 < my_mid) & (my_mid < H+2))
        segs_f      = [s for s, v in zip(segs, valid) if v]
        metric_segs_f = metric_segs[valid]

        # ── Step 5: Collect correspondences ───────────────────────────────────
        all_px:  list[np.ndarray] = []
        all_m:   list[np.ndarray] = []

        # 5a. Sideline points
        for px_pt, m_pt in _sideline_correspondences(h_lines[:1], frame.shape, "top"):
            all_px.append(px_pt); all_m.append(m_pt)
        if len(h_lines) >= 2:
            for px_pt, m_pt in _sideline_correspondences(h_lines[-1:], frame.shape, "bottom"):
                all_px.append(px_pt); all_m.append(m_pt)

        # 5b. Baseline points
        for px_pt, m_pt in _baseline_correspondences(v_lines[:1], frame.shape, "left"):
            all_px.append(px_pt); all_m.append(m_pt)
        if len(v_lines) >= 2:
            for px_pt, m_pt in _baseline_correspondences(v_lines[-1:], frame.shape, "right"):
                all_px.append(px_pt); all_m.append(m_pt)
            features.append("baselines")

        # 5c. Known vertical features (6m, 9m, centre)
        vert_corr = _detect_known_verticals(segs_f, metric_segs_f, fh)
        for px_pt, m_pt in vert_corr:
            all_px.append(px_pt); all_m.append(m_pt)
        if vert_corr:
            features.append(f"known_verticals:{len(vert_corr)}")

        # 5d. 6m arc detection
        arc_results: dict[str, tuple] = {}
        for side in ("left", "right"):
            arc_pts = _collect_arc_candidates(segs_f, metric_segs_f, side,
                                               rough_H, frame.shape)
            if len(arc_pts) < 12:
                continue
            # Expected pixel radius from rough H
            r_expected_px = self._expected_arc_radius_px(rough_H, fw, fh, R6)
            arc = _ransac_circle(arc_pts, tolerance=_ARC_TOL_PX,
                                  iters=500, min_inliers=10)
            if arc is None:
                continue
            cx, cy, r = arc
            # Validate radius: allow 40% slack
            if r_expected_px and not (0.4 * r_expected_px < r < 2.2 * r_expected_px):
                continue
            arc_results[side] = (cx, cy, r)
            features.append(f"6m_arc_{side}")

            arc_corr = _arc_to_correspondences((cx, cy, r), side, rough_H)
            for px_pt, m_pt in arc_corr:
                all_px.append(px_pt); all_m.append(m_pt)

        # ── Step 6: RANSAC refined homography ─────────────────────────────────
        if len(all_px) < 8:
            return CalibResult(False, None,
                               features_found=features + [f"too_few_corr:{len(all_px)}"])

        px_arr = np.array(all_px, np.float32)
        m_arr  = np.array(all_m,  np.float32)

        H_ransac, inlier_mask = cv2.findHomography(
            px_arr, m_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.min_reproj_px,
            maxIters=2000,
            confidence=0.999,
        )

        if H_ransac is None:
            return CalibResult(False, None, features_found=features + ["ransac_failed"])

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        px_in = px_arr[inlier_mask.ravel() == 1]
        m_in  = m_arr [inlier_mask.ravel() == 1]

        # ── Step 7: LM refinement ─────────────────────────────────────────────
        H_refined = _lm_refine(H_ransac, px_in, m_in)

        # ── Step 8: Validate ──────────────────────────────────────────────────
        err = _reproj_error(H_refined, px_in, m_in)
        if err > self.min_reproj_px:
            # Try falling back to RANSAC result
            err_raw = _reproj_error(H_ransac, px_in, m_in)
            H_refined = H_ransac if err_raw < err else H_refined
            err = min(err, err_raw)

        if err > self.min_reproj_px * 1.5:
            return CalibResult(False, None,
                               reproj_error_px=err,
                               features_found=features + [f"high_reproj:{err:.1f}px"])

        if not self._geometric_sanity(H_refined, fw, fh):
            return CalibResult(False, None,
                               reproj_error_px=err,
                               features_found=features + ["geometry_fail"])

        # ── Step 9: Debug frame ───────────────────────────────────────────────
        dbg = None
        if self.debug:
            dbg = _draw_debug(frame, segs, h_lines[:2], v_lines[:2],
                               arc_results,
                               list(zip(all_px, all_m)),
                               H_refined)

        return CalibResult(
            success=True,
            H=H_refined.astype(np.float32),
            reproj_error_px=round(err, 2),
            n_inliers=n_inliers,
            features_found=features,
            debug_frame=dbg,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _partial_H_from_sidelines(
        self,
        h_lines: list,
        v_lines: list,
        fw: int,
        fh: int,
    ) -> Optional[np.ndarray]:
        """
        Compute a rough H from just two sidelines (and optionally one baseline).
        Assumes a top-down view of the full court width.
        """
        if len(h_lines) < 2:
            return None
        top, bot = h_lines[0], h_lines[-1]

        def line_x(rho, theta, y):
            sin_t, cos_t = math.sin(theta), math.cos(theta)
            if abs(cos_t) < 1e-6:
                return rho / cos_t if abs(cos_t) > 1e-9 else fw/2
            return (rho - y * sin_t) / cos_t

        # Estimate corners assuming left/right at x=0 and x=fw
        top_y   = (h_lines[0][0] * math.sin(h_lines[0][1]) +
                   h_lines[0][0] * math.cos(h_lines[0][1]))
        bot_y   = (h_lines[-1][0] * math.sin(h_lines[-1][1]) +
                   h_lines[-1][0] * math.cos(h_lines[-1][1]))
        rho_t, theta_t = h_lines[0]
        rho_b, theta_b = h_lines[-1]

        def pts_on_line(rho, theta):
            ct, st = math.cos(theta), math.sin(theta)
            if abs(st) > 0.1:
                x0, x1 = 0.0, float(fw)
                y0 = (rho - x0*ct) / st
                y1 = (rho - x1*ct) / st
            else:
                y0, y1 = 0.0, float(fh)
                x0 = (rho - y0*st) / ct
                x1 = (rho - y1*st) / ct
            return (x0,y0), (x1,y1)

        tl_side, tr_side = pts_on_line(rho_t, theta_t)
        bl_side, br_side = pts_on_line(rho_b, theta_b)

        # Pick left/right x from vertical lines if available
        if v_lines:
            rho_v, theta_v = v_lines[0]
            # Compute x at top and bottom y
            def vline_x_at_y(y):
                ct, st = math.cos(theta_v), math.sin(theta_v)
                if abs(ct) < 1e-6:
                    return rho_v / ct if abs(ct)>1e-9 else 0.0
                return (rho_v - y*st) / ct
            tl_x = vline_x_at_y(tl_side[1])
            bl_x = vline_x_at_y(bl_side[1])
        else:
            tl_x = 0.0
            bl_x = 0.0

        corners = [
            (tl_x,    tl_side[1]),
            (tr_side[0], tr_side[1]),
            (br_side[0], br_side[1]),
            (bl_x,    bl_side[1]),
        ]
        return _H_from_corners(corners)

    @staticmethod
    def _expected_arc_radius_px(
        H: np.ndarray,
        fw: int,
        fh: int,
        r_m: float,
    ) -> Optional[float]:
        """
        Estimate expected pixel radius of a circle with metric radius r_m
        by back-projecting two points separated by r_m metres.
        """
        try:
            iH = np.linalg.inv(H)
            # Project the centre of the left goal area to pixel space
            centre_m = np.array([[[R6/2, H/2]]], np.float32)
            edge_m   = np.array([[[R6/2 + r_m, H/2]]], np.float32)
            c_px = cv2.perspectiveTransform(centre_m, iH)[0,0]
            e_px = cv2.perspectiveTransform(edge_m,   iH)[0,0]
            return float(np.hypot(c_px[0]-e_px[0], c_px[1]-e_px[1]))
        except Exception:
            return None

    @staticmethod
    def _geometric_sanity(H: np.ndarray, fw: int, fh: int) -> bool:
        """
        Check that H maps a (40×20) metric court to a reasonable pixel quadrilateral:
          • all 4 corners inside frame ± 20%
          • aspect ratio of the quad in [1.0, 4.5]
          • no degenerate / flipped quad
        """
        try:
            iH = np.linalg.inv(H)
            corners_m = np.array([[(0,0),(W,0),(W,H),(0,H)]], np.float32)
            px = cv2.perspectiveTransform(corners_m, iH)[0]

            mx = fw * 0.20; my = fh * 0.20
            for x, y in px:
                if not (-mx < x < fw+mx and -my < y < fh+my):
                    return False

            top_w  = np.linalg.norm(px[1] - px[0])
            bot_w  = np.linalg.norm(px[2] - px[3])
            left_h = np.linalg.norm(px[3] - px[0])
            mean_w = (top_w + bot_w) / 2
            if mean_w < 1 or left_h < 1:
                return False
            ratio = mean_w / left_h
            if not (0.8 < ratio < 5.0):
                return False

            # Check winding order (no flipped quad)
            v1 = px[1] - px[0]
            v2 = px[3] - px[0]
            if np.cross(v1, v2) < 0:
                return False

            return True
        except Exception:
            return False


# ── CourtDetector integration hook ───────────────────────────────────────────

def patch_court_detector():
    """
    Monkey-patch CourtDetector to add set_homography_direct() so AutoCalibrator
    can inject a refined H without going through set_corners().
    """
    try:
        from court_detector import CourtDetector
    except ImportError:
        from src.court_detector import CourtDetector

    if hasattr(CourtDetector, "set_homography_direct"):
        return

    def set_homography_direct(self, H: np.ndarray) -> None:
        """Accept a pre-computed pixel→metre homography directly."""
        self._H  = H.astype(np.float32)
        dst = np.array([(0,0),(W,0),(W,H),(0,H)], np.float32)
        src = np.array([(0,0),(W,0),(W,H),(0,H)], np.float32)
        self._iH, _ = cv2.findHomography(dst, src)   # placeholder
        # Recompute iH properly from H
        self._iH = np.linalg.inv(H).astype(np.float32)
        self._px_cache = None

    CourtDetector.set_homography_direct = set_homography_direct


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    ap = argparse.ArgumentParser(description="Handball court auto-calibrator")
    ap.add_argument("source",  help="Video file, RTSP URL, or integer camera index")
    ap.add_argument("--frames", type=int, default=5,  help="Frames to average (RTSP)")
    ap.add_argument("--skip",   type=int, default=15, help="Frame skip between samples")
    ap.add_argument("--debug",  action="store_true",  help="Show debug visualisation")
    ap.add_argument("--save",   default="",           help="Save result H to JSON file")
    args = ap.parse_args()

    cal = CourtCalibrator(debug=args.debug)

    # Detect input type
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        ret, frame = cap.read(); cap.release()
        if not ret:
            print("Cannot read camera"); sys.exit(1)
        result = cal.calibrate(frame)
    elif src.startswith("rtsp://") or src.startswith("rtsps://"):
        result = cal.calibrate_from_rtsp(src, n_frames=args.frames, frame_skip=args.skip)
    else:
        cap = cv2.VideoCapture(src)
        grabbed, result = 0, None
        best: Optional[CalibResult] = None
        while grabbed < args.frames:
            for _ in range(args.skip):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                break
            r = cal.calibrate(frame)
            grabbed += 1
            if r.success and (best is None or r.reproj_error_px < best.reproj_error_px):
                best = r
        cap.release()
        result = best or CalibResult(False, None)

    # Report
    if result.success:
        print(f"\nCalibration succeeded")
        print(f"  Reprojection error : {result.reproj_error_px:.2f} px")
        print(f"  Inliers            : {result.n_inliers}")
        print(f"  Features detected  : {', '.join(result.features_found)}")
        print(f"  Elapsed            : {result.elapsed_ms:.0f} ms")
        print(f"\n  Homography (pixel→metre):\n{result.H}")

        if args.save:
            import json
            out = {"homography": result.H.tolist(),
                   "inv_homography": np.linalg.inv(result.H).tolist(),
                   "reproj_error_px": result.reproj_error_px,
                   "features": result.features_found}
            Path(args.save).write_text(json.dumps(out, indent=2))
            print(f"\n  Saved → {args.save}")
    else:
        print(f"\nCalibration failed")
        print(f"  Features found: {', '.join(result.features_found) or 'none'}")

    if args.debug and result.debug_frame is not None:
        cv2.imshow("Calibration Debug", result.debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
