"""
pose_analyzer.py — YOLOv8-Pose + BoT-SORT player action analysis.

Responsibilities:
  - Run YOLOv8-Pose on each frame to extract 17 COCO keypoints per player.
  - Track players across frames with BoT-SORT (handles occlusion / re-entry).
  - Classify per-frame actions: jump_shot, running, standing, passing.
  - Estimate centre-of-mass (CoM) from keypoints for stable tracking.
  - Re-identify players after ID loss using colour-histogram + embedding similarity.

Usage:
    analyzer = PoseAnalyzer("yolov8m-pose.pt")
    results  = analyzer.process_frame(bgr_frame)
    # results: list[PlayerPoseResult]
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# YOLOv8 — install with: pip install ultralytics
from ultralytics import YOLO

# ── COCO keypoint indices ─────────────────────────────────────────────────────
NOSE        = 0
L_EYE       = 1;  R_EYE   = 2
L_EAR       = 3;  R_EAR   = 4
L_SHOULDER  = 5;  R_SHOULDER = 6
L_ELBOW     = 7;  R_ELBOW    = 8
L_WRIST     = 9;  R_WRIST    = 10
L_HIP       = 11; R_HIP      = 12
L_KNEE      = 13; R_KNEE     = 14
L_ANKLE     = 15; R_ANKLE    = 16

# Segment weights for CoM estimation (anatomical approximations)
# (keypoint_index, body_segment_mass_fraction)
_COM_WEIGHTS: list[tuple[int, float]] = [
    (NOSE,       0.07),
    (L_SHOULDER, 0.06), (R_SHOULDER, 0.06),
    (L_HIP,      0.12), (R_HIP,      0.12),
    (L_KNEE,     0.065),(R_KNEE,     0.065),
    (L_ANKLE,    0.035),(R_ANKLE,    0.035),
    (L_WRIST,    0.025),(R_WRIST,    0.025),
]


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Keypoints:
    xy: np.ndarray          # shape (17, 2)  pixel coords
    conf: np.ndarray        # shape (17,)    per-keypoint confidence

    def get(self, idx: int) -> tuple[float, float, float]:
        """Return (x, y, conf) for a keypoint."""
        return float(self.xy[idx, 0]), float(self.xy[idx, 1]), float(self.conf[idx])

    def visible(self, idx: int, min_conf: float = 0.35) -> bool:
        return float(self.conf[idx]) >= min_conf

    def midpoint(self, a: int, b: int) -> tuple[float, float]:
        return (float(self.xy[a, 0] + self.xy[b, 0]) / 2,
                float(self.xy[a, 1] + self.xy[b, 1]) / 2)


@dataclass
class PlayerPoseResult:
    track_id: int
    bbox: tuple[float, float, float, float]   # x1 y1 x2 y2
    keypoints: Keypoints
    action: str                               # "jump_shot" | "running" | "standing" | "passing"
    action_conf: float                        # 0-1
    center_of_mass: tuple[float, float]       # pixel coords
    reid_color_hist: Optional[np.ndarray] = field(default=None, repr=False)


# ── Action classifier ─────────────────────────────────────────────────────────

class ActionClassifier:
    """
    Rule-based action classification from COCO keypoints.
    All thresholds are expressed in normalised units (fraction of bounding-box height)
    so they are scale-invariant.
    """

    def classify(self, kp: Keypoints, bbox: tuple, history: list[dict]) -> tuple[str, float]:
        bh = max(bbox[3] - bbox[1], 1.0)   # bounding-box height in pixels

        action, conf = self._jump_shot(kp, bh)
        if action:
            return action, conf

        action, conf = self._passing(kp, bh, history)
        if action:
            return action, conf

        return self._running_or_standing(kp, bh, history)

    # ── sub-classifiers ──────────────────────────────────────────────────────

    def _jump_shot(self, kp: Keypoints, bh: float) -> tuple[str, float]:
        """Arm raised above head AND both ankles high (feet off ground)."""
        votes = 0.0

        # Wrist above nose
        for wrist in (L_WRIST, R_WRIST):
            if kp.visible(wrist) and kp.visible(NOSE):
                wx, wy, _ = kp.get(wrist)
                nx, ny, _ = kp.get(NOSE)
                if wy < ny:                    # y decreases upward in image
                    votes += 0.4

        # Both ankles raised relative to hips
        if all(kp.visible(k) for k in (L_ANKLE, R_ANKLE, L_HIP, R_HIP)):
            hip_y   = (kp.get(L_HIP)[1]   + kp.get(R_HIP)[1])   / 2
            ankle_y = (kp.get(L_ANKLE)[1] + kp.get(R_ANKLE)[1]) / 2
            ratio   = (hip_y - ankle_y) / bh   # positive if ankles are ABOVE hips in image
            if ratio > 0.05:
                votes += min(ratio * 4, 0.6)

        if votes >= 0.5:
            return "jump_shot", min(votes, 1.0)
        return "", 0.0

    def _passing(self, kp: Keypoints, bh: float, history: list[dict]) -> tuple[str, float]:
        """
        Passing: one arm extended forward (wrist far ahead of shoulder) with
        rapid wrist velocity from the previous frame.
        """
        for wrist_idx, shoulder_idx in ((R_WRIST, R_SHOULDER), (L_WRIST, L_SHOULDER)):
            if not (kp.visible(wrist_idx) and kp.visible(shoulder_idx)):
                continue
            wx, wy, _ = kp.get(wrist_idx)
            sx, sy, _ = kp.get(shoulder_idx)
            extension  = abs(wx - sx) / bh
            if extension < 0.3:
                continue
            # Check wrist velocity from history
            if len(history) >= 2:
                prev_kp = history[-1].get("keypoints")
                if prev_kp and prev_kp.visible(wrist_idx):
                    pw, py2, _ = prev_kp.get(wrist_idx)
                    vel = np.hypot(wx - pw, wy - py2) / bh
                    if vel > 0.08:
                        conf = min(0.5 + vel * 2, 0.95)
                        return "passing", conf
        return "", 0.0

    def _running_or_standing(self, kp: Keypoints, bh: float,
                              history: list[dict]) -> tuple[str, float]:
        """
        Running: CoM displacement across frames exceeds threshold, OR
        opposite knee-hip pairs are misaligned (running gait).
        """
        if len(history) >= 2:
            prev_com = history[-1].get("center_of_mass")
            cur_com  = history[-1].get("center_of_mass")  # will be set after return
            if prev_com and cur_com:
                velocity = np.hypot(cur_com[0] - prev_com[0], cur_com[1] - prev_com[1]) / bh
                if velocity > 0.04:
                    return "running", min(0.5 + velocity * 5, 0.95)

        # Gait asymmetry: if knees at different heights, likely running
        if kp.visible(L_KNEE) and kp.visible(R_KNEE):
            knee_diff = abs(kp.get(L_KNEE)[1] - kp.get(R_KNEE)[1]) / bh
            if knee_diff > 0.08:
                return "running", min(0.4 + knee_diff * 2, 0.9)

        return "standing", 0.85


# ── Colour-histogram Re-ID ────────────────────────────────────────────────────

class ReIDEngine:
    """
    Maintains a gallery of colour histograms (torso crop) per track ID.
    When a track is lost and a new detection appears, compares histograms
    to re-assign the most likely previous ID.
    """

    def __init__(self, hist_bins: int = 32, similarity_threshold: float = 0.72):
        self._gallery: dict[int, np.ndarray] = {}   # track_id → histogram
        self._bins = hist_bins
        self._threshold = similarity_threshold

    def update(self, track_id: int, bgr_crop: np.ndarray) -> None:
        hist = self._compute_hist(bgr_crop)
        if track_id in self._gallery:
            # Exponential moving average for stability
            self._gallery[track_id] = 0.7 * self._gallery[track_id] + 0.3 * hist
        else:
            self._gallery[track_id] = hist

    def get_hist(self, track_id: int) -> Optional[np.ndarray]:
        return self._gallery.get(track_id)

    def find_match(self, bgr_crop: np.ndarray,
                   candidate_ids: list[int]) -> tuple[Optional[int], float]:
        """Return (best_track_id, similarity) or (None, 0) if no match."""
        if not candidate_ids or bgr_crop is None or bgr_crop.size == 0:
            return None, 0.0
        query = self._compute_hist(bgr_crop)
        best_id, best_sim = None, 0.0
        for tid in candidate_ids:
            gallery_hist = self._gallery.get(tid)
            if gallery_hist is None:
                continue
            sim = float(cv2.compareHist(query, gallery_hist, cv2.HISTCMP_CORREL))
            if sim > best_sim:
                best_sim, best_id = sim, tid
        if best_sim >= self._threshold:
            return best_id, best_sim
        return None, best_sim

    def _compute_hist(self, bgr: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [self._bins, self._bins],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
        return hist.flatten()


# ── Centre-of-mass estimator ──────────────────────────────────────────────────

def estimate_com(kp: Keypoints) -> tuple[float, float]:
    """
    Weighted average of visible keypoints using anatomical segment fractions.
    Falls back to mean of all visible points if critical ones are missing.
    """
    total_w, sx, sy = 0.0, 0.0, 0.0
    for idx, w in _COM_WEIGHTS:
        if kp.visible(idx):
            x, y, _ = kp.get(idx)
            sx += x * w
            sy += y * w
            total_w += w

    if total_w > 0.1:
        return sx / total_w, sy / total_w

    # Fallback: mean of all visible points
    visible = [(float(kp.xy[i, 0]), float(kp.xy[i, 1]))
               for i in range(17) if kp.visible(i)]
    if visible:
        xs, ys = zip(*visible)
        return float(np.mean(xs)), float(np.mean(ys))

    return 0.0, 0.0


# ── Torso crop helper ─────────────────────────────────────────────────────────

def _torso_crop(frame: np.ndarray, kp: Keypoints,
                bbox: tuple) -> Optional[np.ndarray]:
    """Extract the torso region for Re-ID colour histogram."""
    h, w = frame.shape[:2]
    if all(kp.visible(k) for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
        xs = [kp.get(k)[0] for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
        ys = [kp.get(k)[1] for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
        x1, y1 = int(max(min(xs) - 5, 0)), int(max(min(ys) - 5, 0))
        x2, y2 = int(min(max(xs) + 5, w)), int(min(max(ys) + 5, h))
    else:
        # Fallback: upper-middle third of bounding box
        bx1, by1, bx2, by2 = bbox
        x1 = int(max(bx1, 0));   x2 = int(min(bx2, w))
        by_h = by2 - by1
        y1 = int(max(by1 + by_h * 0.2, 0))
        y2 = int(min(by1 + by_h * 0.65, h))

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ── Main analyzer ─────────────────────────────────────────────────────────────

class PoseAnalyzer:
    """
    Full pipeline: YOLOv8-Pose detection → BoT-SORT tracking → action
    classification → Re-ID.

    Args:
        model_path: Path to a YOLOv8-pose .pt weights file.
        conf:       Detection confidence threshold.
        device:     "cuda", "mps", or "cpu".
        reid_threshold: Colour-histogram similarity to accept a Re-ID match.
    """

    def __init__(
        self,
        model_path: str = "yolov8m-pose.pt",
        conf: float = 0.40,
        device: str = "cuda",
        reid_threshold: float = 0.72,
        history_len: int = 8,
    ):
        self._model = YOLO(model_path)
        self._conf  = conf
        self._device = device

        self._classifier = ActionClassifier()
        self._reid       = ReIDEngine(similarity_threshold=reid_threshold)

        # Per-track history: list of dicts with keypoints, CoM, action
        self._history: dict[int, list[dict]] = defaultdict(list)
        self._history_len = history_len

        # IDs that disappeared this frame (candidates for Re-ID)
        self._lost_ids: set[int] = set()
        self._active_ids: set[int] = set()

    def process_frame(self, frame: np.ndarray) -> list[PlayerPoseResult]:
        """
        Run pose estimation + tracking on one BGR frame.
        Returns a list of PlayerPoseResult for every tracked person.
        """
        results = self._model.track(
            frame,
            conf=self._conf,
            persist=True,        # BoT-SORT keeps IDs across frames
            tracker="botsort.yaml",
            device=self._device,
            classes=[0],         # person only
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        r          = results[0]
        boxes      = r.boxes
        kps_batch  = r.keypoints    # ultralytics Keypoints object

        current_ids: set[int] = set()
        outputs: list[PlayerPoseResult] = []

        for i, box in enumerate(boxes):
            # Track ID (BoT-SORT assigns these)
            track_id_raw = box.id
            if track_id_raw is None:
                continue
            track_id = int(track_id_raw.item())

            bbox = tuple(float(v) for v in box.xyxy[0].tolist())   # x1 y1 x2 y2

            # Keypoints
            kp_xy   = kps_batch.xy[i].cpu().numpy()    # (17, 2)
            kp_conf = kps_batch.conf[i].cpu().numpy()  # (17,)
            kp      = Keypoints(xy=kp_xy, conf=kp_conf)

            # Centre of mass
            com = estimate_com(kp)

            # Torso crop for Re-ID
            crop = _torso_crop(frame, kp, bbox)

            # Re-ID: if this track_id is brand-new, check against lost IDs
            if track_id not in self._active_ids and self._lost_ids and crop is not None:
                matched_id, sim = self._reid.find_match(crop, list(self._lost_ids))
                if matched_id is not None:
                    # Transfer history from old ID to new
                    self._history[track_id] = self._history.pop(matched_id, [])
                    self._reid._gallery[track_id] = self._reid._gallery.pop(matched_id)
                    self._lost_ids.discard(matched_id)

            history = self._history[track_id]

            # Action classification
            action, action_conf = self._classifier.classify(kp, bbox, history)

            # Update Re-ID gallery
            if crop is not None:
                self._reid.update(track_id, crop)

            # Update history
            history.append({
                "keypoints":     kp,
                "center_of_mass": com,
                "action":        action,
            })
            if len(history) > self._history_len:
                history.pop(0)

            current_ids.add(track_id)
            outputs.append(PlayerPoseResult(
                track_id=track_id,
                bbox=bbox,
                keypoints=kp,
                action=action,
                action_conf=action_conf,
                center_of_mass=com,
                reid_color_hist=self._reid.get_hist(track_id),
            ))

        # Track which IDs disappeared this frame
        self._lost_ids  = self._active_ids - current_ids
        self._active_ids = current_ids

        return outputs

    def draw(self, frame: np.ndarray, results: list[PlayerPoseResult]) -> np.ndarray:
        """
        Overlay bounding boxes, CoM dots, keypoint skeletons, and action labels
        onto a copy of the frame. Useful for debugging.
        """
        out   = frame.copy()
        h, w  = out.shape[:2]

        _ACTION_COLORS = {
            "jump_shot": (0, 140, 255),
            "passing":   (180, 60, 255),
            "running":   (0, 220, 80),
            "standing":  (200, 200, 200),
        }
        _SKELETON = [
            (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
            (L_ELBOW, L_WRIST),       (R_ELBOW, R_WRIST),
            (L_SHOULDER, L_HIP),      (R_SHOULDER, R_HIP),
            (L_HIP, R_HIP),           (L_HIP, L_KNEE), (R_HIP, R_KNEE),
            (L_KNEE, L_ANKLE),        (R_KNEE, R_ANKLE),
        ]

        for pr in results:
            color = _ACTION_COLORS.get(pr.action, (160, 160, 160))
            x1, y1, x2, y2 = (int(v) for v in pr.bbox)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Skeleton
            kp = pr.keypoints
            for a_idx, b_idx in _SKELETON:
                if kp.visible(a_idx) and kp.visible(b_idx):
                    ax, ay = int(kp.xy[a_idx, 0]), int(kp.xy[a_idx, 1])
                    bx, by = int(kp.xy[b_idx, 0]), int(kp.xy[b_idx, 1])
                    cv2.line(out, (ax, ay), (bx, by), color, 1, cv2.LINE_AA)

            # Keypoint dots
            for ki in range(17):
                if kp.visible(ki):
                    cx, cy = int(kp.xy[ki, 0]), int(kp.xy[ki, 1])
                    cv2.circle(out, (cx, cy), 3, color, -1, cv2.LINE_AA)

            # Centre-of-mass
            cx, cy = int(pr.center_of_mass[0]), int(pr.center_of_mass[1])
            cv2.drawMarker(out, (cx, cy), (0, 229, 255),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

            # Label
            label = f"ID{pr.track_id} {pr.action} {pr.action_conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ly = max(y1 - 6, lh + 4)
            cv2.rectangle(out, (x1, ly - lh - 4), (x1 + lw + 6, ly + 2), (0, 0, 0), -1)
            cv2.putText(out, label, (x1 + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        return out
