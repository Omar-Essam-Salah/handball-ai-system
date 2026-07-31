"""
Court Detector — YOLO-Pose keypoint-based court calibration + full IHF overlay.

Detection pipeline  (background thread)
───────────────────────────────────────
1. YOLOv8-Pose detects court keypoints each frame via ``update(frame)``.
2. Keypoints with confidence ≥ kp_conf_threshold are matched to their
   known IHF metre coordinates (see COURT_KP_METER_POS).
3. cv2.findHomography (RANSAC) computes the pixel→metre transform.
4. If fewer than min_kp_for_homography keypoints are visible, the last
   known homography is retained (occlusion / camera-move fallback).

Keypoint convention — see COURT_KP_METER_POS below.
Manual calibration fallback — detector.set_corners([TL, TR, BR, BL]).

Quick-start
───────────
    det = CourtDetector(model_path="models/court_pose.pt")

    # Background thread:
    det.update(frame)                   # YOLO inference → updates H

    # Main thread (every frame):
    frame = det.draw_overlay(frame)     # IHF overlay
    m_pos = det.pixel_to_meter(px, py)  # pixel → metres
"""

import math
import json
import threading
import cv2
import numpy as np
from pathlib import Path

# ── IHF geometry constants (metres) ──────────────────────────────────────────
W   = 40.0   # court length  (x-axis, 0 → 40)
H   = 20.0   # court width   (y-axis, 0 → 20)
CX  = W / 2  # 20.0
CY  = H / 2  # 10.0

_LP  = (0.0,   8.5)
_LPu = (0.0,  11.5)
_RP  = (40.0,  8.5)
_RPu = (40.0, 11.5)

R6 = 6.0
R9 = 9.0

CALIB_PATH = Path(__file__).parent.parent / "config" / "court_calibration.json"

# ── Keypoint → IHF metre position mapping ────────────────────────────────────
# Maps YOLO-Pose keypoint index → (x_metres, y_metres) in IHF coordinates.
# These indices MUST match the keypoint order in your training dataset.
# See the "How to define keypoints" section at the bottom of this file.
#
# Minimal 4-point config (required minimum):
COURT_KP_METER_POS: dict[int, tuple[float, float]] = {
    0: (0.0,   0.0),   # Top-Left     corner
    1: (40.0,  0.0),   # Top-Right    corner
    2: (40.0, 20.0),   # Bottom-Right corner
    3: (0.0,  20.0),   # Bottom-Left  corner
    # Recommended additions for robustness under partial occlusion:
    # 4: (20.0,  0.0),   # Centre-line × top boundary
    # 5: (20.0, 20.0),   # Centre-line × bottom boundary
    # 6: ( 6.0,  8.5),   # Left 6m arc right-end  (lower)
    # 7: ( 6.0, 11.5),   # Left 6m arc right-end  (upper)
    # 8: (34.0,  8.5),   # Right 6m arc left-end  (lower)
    # 9: (34.0, 11.5),   # Right 6m arc left-end  (upper)
}

MIN_KP_FOR_HOMOGRAPHY = 4    # minimum visible keypoints to solve homography
KP_CONF_THRESHOLD     = 0.5  # YOLO keypoint confidence cutoff
RANSAC_REPROJ_THRESH  = 5.0  # pixels — RANSAC reprojection tolerance


# ── Arc / geometry helpers ────────────────────────────────────────────────────

def _arc(cx: float, cy: float, r: float,
         a_start: float, a_end: float, n: int = 40) -> np.ndarray:
    """Sample n points on a circle arc (angles in degrees, CCW).
    Returns (n, 2) float32 array in metres."""
    angles = np.linspace(math.radians(a_start), math.radians(a_end), n)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _line_pts(*pts) -> np.ndarray:
    """Convert (x, y) tuples to (N, 2) float32 array."""
    return np.array(pts, dtype=np.float32)


def _9m_start_angle(cy_post: float, side: str) -> float:
    """Angle where the 9m arc hits the sideline (y=0 or y=20)."""
    sideline = 0.0 if cy_post < CY else H
    ratio = np.clip((sideline - cy_post) / R9, -1.0, 1.0)
    base = math.degrees(math.asin(ratio))
    return base if side == "left" else 180.0 - base


def build_court_geometry() -> list[dict]:
    """
    Return every IHF court feature as a dict:
        pts      : np.ndarray (N,2) metres
        color    : (B,G,R)
        thick    : line thickness
        dashed   : True → dashed polyline
        fill     : True → filled polygon (semi-transparent)
        fill_alpha: opacity for filled polygons
    """
    feats = []

    # ── Court boundary ────────────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts((0,0),(W,0),(W,H),(0,H),(0,0)),
                      color=(255,255,255), thick=3, dashed=False))

    # ── Centre line ───────────────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts((CX,0),(CX,H)),
                      color=(255,255,255), thick=2, dashed=False))

    # ── Substitution marks ────────────────────────────────────────────────────
    sub = 4.5
    for y in (0, H):
        sign = 1.0 if y == 0 else -1.0
        for sx in (CX - sub, CX + sub):
            feats.append(dict(pts=_line_pts((sx, y), (sx, y + sign*0.5)),
                              color=(255,255,255), thick=2, dashed=False))

    # ── Left goal ─────────────────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts(_LP, _LPu),
                      color=(255,255,255), thick=4, dashed=False))

    # ── Left 6m area (solid) ──────────────────────────────────────────────────
    a1  = _arc(0,  8.5, R6, -90,   0)
    seg = _line_pts((6, 8.5), (6, 11.5))
    a2  = _arc(0, 11.5, R6,   0,  90)
    line6L = np.concatenate([a1, seg, a2])
    feats.append(dict(pts=line6L, color=(0,215,255), thick=2, dashed=False))
    fill6L = np.concatenate([_line_pts((0,2.5),(0,8.5)), a1, seg, a2,
                              _line_pts((0,17.5),(0,11.5))])
    feats.append(dict(pts=fill6L, color=(0,180,220), thick=0,
                      dashed=False, fill=True, fill_alpha=0.18))

    # ── Left 9m line (dashed) ─────────────────────────────────────────────────
    a3   = _arc(0,  8.5, R9, _9m_start_angle(8.5,  "left"), 0)
    seg9 = _line_pts((9, 8.5), (9, 11.5))
    a4   = _arc(0, 11.5, R9, 0, _9m_start_angle(11.5, "left"))
    for chunk in (a3, seg9, a4):
        feats.append(dict(pts=chunk, color=(255,140,0), thick=2, dashed=True))

    # ── Left 7m penalty spot ──────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts((7, CY-0.5),(7, CY+0.5)),
                      color=(0,0,255), thick=3, dashed=False))

    # ── Left 4m goalkeeper line ───────────────────────────────────────────────
    feats.append(dict(pts=_line_pts((4, 8.5),(4, 11.5)),
                      color=(255,255,255), thick=2, dashed=False))

    # ── Right goal ────────────────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts(_RP, _RPu),
                      color=(255,255,255), thick=4, dashed=False))

    # ── Right 6m area ─────────────────────────────────────────────────────────
    ra1  = _arc(40,  8.5, R6, 180, 270)
    rseg = _line_pts((34, 8.5),(34, 11.5))
    ra2  = _arc(40, 11.5, R6,  90, 180)
    line6R = np.concatenate([ra1[::-1], rseg, ra2[::-1]])
    feats.append(dict(pts=line6R, color=(0,215,255), thick=2, dashed=False))
    fill6R = np.concatenate([ra1[::-1], rseg, ra2[::-1],
                              _line_pts((40,17.5),(40,11.5)),
                              _line_pts((40,8.5),(40,2.5))])
    feats.append(dict(pts=fill6R, color=(0,180,220), thick=0,
                      dashed=False, fill=True, fill_alpha=0.18))

    # ── Right 9m line (dashed) ────────────────────────────────────────────────
    ra3   = _arc(40,  8.5, R9, 180, _9m_start_angle(8.5,  "right"))
    rseg9 = _line_pts((31, 8.5),(31, 11.5))
    ra4   = _arc(40, 11.5, R9, _9m_start_angle(11.5, "right"), 180)
    for chunk in (ra3, rseg9, ra4):
        feats.append(dict(pts=chunk, color=(255,140,0), thick=2, dashed=True))

    # ── Right 7m penalty spot ─────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts((33, CY-0.5),(33, CY+0.5)),
                      color=(0,0,255), thick=3, dashed=False))

    # ── Right 4m line ─────────────────────────────────────────────────────────
    feats.append(dict(pts=_line_pts((36, 8.5),(36, 11.5)),
                      color=(255,255,255), thick=2, dashed=False))

    # ── Goal nets (1m deep visual guide) ─────────────────────────────────────
    for gx, depth in ((0, -1.0), (W, +1.0)):
        net = _line_pts((gx,8.5),(gx+depth,8.5),(gx+depth,11.5),(gx,11.5))
        feats.append(dict(pts=net, color=(60,255,60), thick=2,
                          dashed=False, fill=True, fill_alpha=0.25))

    return feats


# ── Main class ────────────────────────────────────────────────────────────────

class CourtDetector:
    """
    Detects the handball court using YOLOv8-Pose keypoints and renders a
    full IHF overlay.  Designed for a fixed producer/consumer threading model:

        Background thread → det.update(frame)       # YOLO inference
        Main thread       → det.draw_overlay(frame) # overlay rendering
    """

    def __init__(self,
                 model_path: str | Path | None = None,
                 calib_path: Path = CALIB_PATH,
                 kp_conf_threshold: float = KP_CONF_THRESHOLD,
                 min_kp: int = MIN_KP_FOR_HOMOGRAPHY):
        """
        Parameters
        ──────────
        model_path        : path to a YOLOv8-Pose .pt file.  If None, YOLO
                            detection is disabled (manual calibration only).
        calib_path        : where to persist the last-good homography as JSON.
        kp_conf_threshold : discard keypoints below this YOLO confidence.
        min_kp            : minimum visible keypoints required to recompute H.
        """
        self._lock        = threading.RLock()
        self._H:  np.ndarray | None = None   # pixel → metre
        self._iH: np.ndarray | None = None   # metre → pixel
        self._H_version   = 0                # incremented on every H update
        self._px_cache_ver = -1
        self._px_cache:  list | None = None

        self._calib_path  = calib_path
        self._kp_conf     = kp_conf_threshold
        self._min_kp      = min_kp
        self._geometry    = build_court_geometry()
        self._model       = None

        if model_path is not None:
            self._load_model(Path(model_path))

        self._load()   # restore persisted homography

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(str(path))
            # Warm-up: pre-load weights into memory on a blank frame
            dummy = np.zeros((320, 320, 3), np.uint8)
            self._model(dummy, verbose=False)
            print(f"[CourtDetector] YOLO model loaded: {path.name}")
        except ImportError:
            print("[CourtDetector] 'ultralytics' not installed — YOLO disabled.")
        except Exception as e:
            print(f"[CourtDetector] Model load failed: {e}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._H is not None

    # ── YOLO-Pose update (call from background thread) ────────────────────────

    def update(self, frame: np.ndarray) -> bool:
        """
        Run one YOLO-Pose inference pass and update the homography.

        Returns True  → new homography computed and stored.
        Returns False → fewer than min_kp keypoints visible; last H retained.

        Thread-safe: acquires the internal lock only during the H swap.
        """
        if self._model is None:
            return False

        try:
            results = self._model(frame, verbose=False)
        except Exception as e:
            print(f"[CourtDetector] Inference error: {e}")
            return False

        if not results or results[0].keypoints is None:
            return False

        kp_data = results[0].keypoints.data  # (N_detections, K, 3)
        if len(kp_data) == 0:
            return False

        # Use the first (highest-confidence) detection — the court object
        court_kps = kp_data[0].cpu().numpy()  # (K, 3): [x, y, conf]

        src_pts: list[list[float]] = []   # pixel positions
        dst_pts: list[list[float]] = []   # corresponding IHF metres

        for kp_idx, metre_pos in COURT_KP_METER_POS.items():
            if kp_idx >= len(court_kps):
                continue
            x, y, conf = court_kps[kp_idx]
            if conf >= self._kp_conf:
                src_pts.append([float(x), float(y)])
                dst_pts.append([float(metre_pos[0]), float(metre_pos[1])])

        if len(src_pts) < self._min_kp:
            # Occlusion fallback: keep last known homography
            return False

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
        if H is None:
            return False

        try:
            iH = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return False

        with self._lock:
            self._H  = H.astype(np.float32)
            self._iH = iH.astype(np.float32)
            self._H_version += 1

        return True

    # ── Manual calibration ────────────────────────────────────────────────────

    def set_corners(self, pixel_pts: list) -> bool:
        """
        Manually set calibration from 4 pixel points.
        pixel_pts: [TL, TR, BR, BL] as (x, y) tuples.
        """
        if len(pixel_pts) != 4:
            return False
        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array([(0,0),(W,0),(W,H),(0,H)], dtype=np.float32)
        H,  _ = cv2.findHomography(src, dst)
        iH, _ = cv2.findHomography(dst, src)
        if H is None or iH is None:
            return False
        with self._lock:
            self._H  = H.astype(np.float32)
            self._iH = iH.astype(np.float32)
            self._H_version += 1
        self._save()
        return True

    def set_manual_corners(self, pixel_pts: list) -> bool:
        """Alias for set_corners."""
        return self.set_corners(pixel_pts)

    def set_homography_direct(self, H: np.ndarray) -> None:
        """
        Accept a pre-computed pixel→metre homography (e.g. from CourtCalibrator).
        """
        iH = np.linalg.inv(H)
        with self._lock:
            self._H  = H.astype(np.float32)
            self._iH = iH.astype(np.float32)
            self._H_version += 1
        self._save()

    def clear_calibration(self) -> None:
        """Reset to uncalibrated state and delete the saved JSON."""
        with self._lock:
            self._H  = None
            self._iH = None
            self._H_version += 1
        try:
            if self._calib_path.exists():
                self._calib_path.unlink()
                print("[CourtDetector] Calibration cleared.")
        except Exception as e:
            print(f"[CourtDetector] Clear failed: {e}")

    def auto_calibrate(self, frame: np.ndarray) -> bool:
        """
        Compatibility shim: delegates to update() when a YOLO model is loaded.
        Returns False immediately if no model is available (use set_corners).
        """
        return self.update(frame)

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def pixel_to_meter(self, x: float, y: float) -> tuple[float, float] | None:
        with self._lock:
            if self._H is None:
                return None
            H = self._H
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def meter_to_pixel(self, xm: float, ym: float) -> tuple[int, int] | None:
        with self._lock:
            if self._iH is None:
                return None
            iH = self._iH
        pt = cv2.perspectiveTransform(np.array([[[xm, ym]]], np.float32), iH)
        return int(pt[0, 0, 0]), int(pt[0, 0, 1])

    def is_goal(self, x_m: float, y_m: float) -> str | None:
        """Return 'left' | 'right' | None."""
        if 8.5 <= y_m <= 11.5:
            if x_m <= 0.3:
                return "left"
            if x_m >= W - 0.3:
                return "right"
        return None

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """
        Render the full IHF court overlay onto frame.
        Safe to call from the main thread while update() runs in another.
        """
        with self._lock:
            calibrated = self._H is not None
            cur_ver    = self._H_version
            iH         = self._iH

        if not calibrated:
            return frame

        px_feats = self._build_pixel_cache(frame.shape, cur_ver, iH)
        out = frame.copy()

        # Pass 1: filled zones (alpha-blended)
        fill_layer = out.copy()
        for feat, px_pts in zip(self._geometry, px_feats):
            if feat.get("fill") and px_pts is not None and len(px_pts) >= 3:
                poly = px_pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(fill_layer, [poly], feat["color"])
        cv2.addWeighted(fill_layer, alpha, out, 1 - alpha, 0, out)

        # Pass 2: solid / dashed lines
        for feat, px_pts in zip(self._geometry, px_feats):
            if feat.get("fill") or px_pts is None or len(px_pts) < 2:
                continue
            pts_i = px_pts.astype(np.int32)
            if feat.get("dashed"):
                _draw_dashed(out, pts_i, feat["color"], feat["thick"])
            else:
                cv2.polylines(out, [pts_i.reshape(-1, 1, 2)],
                              isClosed=False, color=feat["color"],
                              thickness=feat["thick"], lineType=cv2.LINE_AA)

        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        with self._lock:
            if self._iH is None:
                return
            H_list  = self._H.tolist()
            iH_list = self._iH.tolist()
        try:
            self._calib_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._calib_path, "w") as f:
                json.dump({"homography": H_list, "inv_homography": iH_list}, f)
        except Exception as e:
            print(f"[CourtDetector] Save failed: {e}")

    def _load(self) -> None:
        if not self._calib_path.exists():
            return
        try:
            d = json.loads(self._calib_path.read_text())
            H  = np.array(d["homography"],     np.float32) if "homography"     in d else None
            iH = np.array(d["inv_homography"], np.float32) if "inv_homography" in d else None
            if H is not None and iH is None:
                iH = np.linalg.inv(H).astype(np.float32)
            with self._lock:
                self._H  = H
                self._iH = iH
                self._H_version += 1
            print("[CourtDetector] Calibration loaded from file.")
        except Exception as e:
            print(f"[CourtDetector] Load failed: {e}")

    # ── Pixel-space cache ─────────────────────────────────────────────────────

    def _build_pixel_cache(self, shape: tuple,
                           version: int, iH: np.ndarray) -> list:
        """
        Transform metre geometry → pixel geometry and cache by H version.
        The cache is invalidated whenever the homography changes, so a dynamic
        (moving) camera pays one transform per frame — still < 1 ms.
        """
        if self._px_cache is not None and self._px_cache_ver == version:
            return self._px_cache

        fh, fw = shape[:2]
        cache = []
        for feat in self._geometry:
            pts_m  = feat["pts"].reshape(-1, 1, 2)
            pts_px = cv2.perspectiveTransform(pts_m, iH)
            if pts_px is None:
                cache.append(None)
                continue
            pts_px = np.clip(pts_px.reshape(-1, 2),
                             [-fw * 0.1, -fh * 0.1],
                             [ fw * 1.1,  fh * 1.1])
            cache.append(pts_px.astype(np.float32))

        self._px_cache     = cache
        self._px_cache_ver = version
        return cache


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_dashed(img: np.ndarray, pts: np.ndarray,
                 color: tuple, thick: int,
                 dash: int = 12, gap: int = 8) -> None:
    """Draw a dashed polyline through a sequence of integer pixel points."""
    if len(pts) < 2:
        return
    carry   = 0.0
    drawing = True
    for i in range(len(pts) - 1):
        p0 = pts[i].astype(float)
        p1 = pts[i + 1].astype(float)
        seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if seg_len < 1:
            continue
        dx, dy = (p1[0] - p0[0]) / seg_len, (p1[1] - p0[1]) / seg_len
        t = carry
        while t < seg_len:
            block = dash if drawing else gap
            t_end = min(t + block, seg_len)
            if drawing:
                s = (int(p0[0] + t     * dx), int(p0[1] + t     * dy))
                e = (int(p0[0] + t_end * dx), int(p0[1] + t_end * dy))
                cv2.line(img, s, e, color, thick, cv2.LINE_AA)
            t       = t_end
            drawing = not drawing
        carry = t - seg_len


def _sort_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL]."""
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]])


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO DEFINE KEYPOINTS FOR YOUR YOLO-POSE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
#
# Step 1 — Annotate your dataset
# ───────────────────────────────
#   Use Roboflow, CVAT, or LabelStudio.  For each court image, mark the
#   keypoints in a FIXED order that matches COURT_KP_METER_POS above.
#   The default 4-point order is:
#       KP 0 → Top-Left     corner  (0 m,  0 m)
#       KP 1 → Top-Right    corner  (40 m, 0 m)
#       KP 2 → Bottom-Right corner  (40 m, 20 m)
#       KP 3 → Bottom-Left  corner  (0 m,  20 m)
#   "Top" = the side closer to the camera in your typical view.
#   Always annotate keypoints in the same spatial order regardless of
#   camera angle — the model must learn consistent numbering.
#
# Step 2 — dataset.yaml (YOLO-Pose format)
# ─────────────────────────────────────────
#   kpt_shape: [4, 3]   # [num_keypoints, (x, y, visibility)]
#   flip_idx: [1, 0, 3, 2]   # mirror mapping for horizontal flips
#              # KP0↔KP1, KP2↔KP3 (left-right corners swap on h-flip)
#
# Step 3 — Training
# ──────────────────
#   from ultralytics import YOLO
#   model = YOLO("yolov8n-pose.pt")          # start from nano weights
#   model.train(data="court_pose.yaml", epochs=100, imgsz=640, batch=16)
#
# Step 4 — Verify keypoint order at inference
# ────────────────────────────────────────────
#   results = model(test_frame)
#   kps = results[0].keypoints.data[0]       # (4, 3)
#   for i, (x, y, conf) in enumerate(kps):
#       cv2.circle(test_frame, (int(x), int(y)), 8, (0,255,0), -1)
#       cv2.putText(test_frame, str(i), (int(x)+5, int(y)-5),
#                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#   # Numbers must match the corners above. If not, adjust COURT_KP_METER_POS.
#
# Step 5 — Add more keypoints for robustness (optional but recommended)
# ──────────────────────────────────────────────────────────────────────
#   With 6+ keypoints, RANSAC can tolerate 1-2 occluded points per frame.
#   Good candidates: centre-line endpoints, goal-post bases, 6m arc tangent
#   points.  Uncomment the entries in COURT_KP_METER_POS and re-annotate.
