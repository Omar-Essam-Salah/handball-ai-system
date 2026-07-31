"""
Auto-calibrating team color detector.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

_X_MARGIN  = 0.20   
_Y_TOP     = 0.10   
_Y_BOTTOM  = 0.50   

# تم تقليل المتطلبات لتسريع الإخفاء
MIN_CROPS_TO_CALIBRATE = 15    
DOMINANT_K             = 3     
TEAM_K                 = 3     

# تم التخفيف جداً هنا لكي يقبل أي قميص حتى لو أبيض/أسود
_MIN_BRIGHTNESS_L      = 10    
_MIN_COLOR_SPREAD      = 1.5   
_MIN_CLUSTER_DISTANCE  = 3     

_FALLBACK_COLORS = {
    "team_a":    (220, 60,  60),    
    "team_b":    (0,   140, 220),   
    "goalkeeper":(0,   200, 200),   
}

@dataclass
class TeamProfile:
    label:      str            
    center_lab: np.ndarray     
    color_bgr:  tuple          
    count:      int = 0        


class AutoColorDetector:
    def __init__(self, min_crops: int = MIN_CROPS_TO_CALIBRATE):
        self.min_crops        = min_crops
        self._color_buffer: list[np.ndarray] = []   
        self.profiles:      list[TeamProfile] = []
        self.is_calibrated    = False
        self.calibration_ok   = False   

    @property
    def calibration_progress(self) -> float:
        return min(1.0, len(self._color_buffer) / max(1, self.min_crops))

    def feed(self, frame: np.ndarray, tracks: list) -> None:
        if self.is_calibrated:
            return

        for t in tracks:
            if getattr(t, 'is_ball', False):
                continue
            crop = self._torso_crop(frame, t)
            if crop is None:
                continue
            dom = self._dominant_color_lab(crop)
            if dom is not None:
                self._color_buffer.append(dom)

        if len(self._color_buffer) >= self.min_crops:
            self._calibrate()

    def classify(self, frame: np.ndarray, track) -> str:
        if not self.is_calibrated: return "unknown"
        crop = self._torso_crop(frame, track)
        if crop is None: return "unknown"
        dom = self._dominant_color_lab(crop)
        if dom is None: return "unknown"

        dists = [np.linalg.norm(dom.astype(float) - p.center_lab.astype(float)) for p in self.profiles]
        return self.profiles[int(np.argmin(dists))].label

    def team_bgr_colors(self) -> dict[str, tuple]:
        return {p.label: p.color_bgr for p in self.profiles}

    def recalibrate(self) -> None:
        self._color_buffer.clear()
        self.profiles.clear()
        self.is_calibrated  = False
        self.calibration_ok = False

    # ── الإنهاء الإجباري لتخطي التعليق ──
    def force_calibration(self) -> None:
        if not self.is_calibrated:
            if len(self._color_buffer) >= 3:
                self._calibrate()
            else:
                self._use_fallback_colors("Forced timeout")

    def _torso_crop(self, frame: np.ndarray, track) -> np.ndarray | None:
        fh, fw = frame.shape[:2]
        bx1, bx2 = int(max(0, track.x1)), int(min(fw, track.x2))
        by1, by2 = int(max(0, track.y1)), int(min(fh, track.y2))
        bw, bh  = bx2 - bx1, by2 - by1

        if bw < 8 or bh < 12: return None

        tx1, tx2 = bx1 + int(bw * _X_MARGIN), bx2 - int(bw * _X_MARGIN)
        ty1, ty2 = by1 + int(bh * _Y_TOP), by1 + int(bh * _Y_BOTTOM)

        if tx2 <= tx1 or ty2 <= ty1: return None
        crop = frame[ty1:ty2, tx1:tx2]
        return crop if crop.size > 0 else None

    @staticmethod
    def _enhance_crop(crop_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _dominant_color_lab(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        enhanced = self._enhance_crop(crop_bgr)
        lab      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        pixels   = lab.reshape(-1, 3).astype(np.float32)

        if len(pixels) < DOMINANT_K * 3: return None
        if float(np.mean(pixels[:, 0])) < _MIN_BRIGHTNESS_L: return None
        if float(np.std(pixels[:, 1:])) < _MIN_COLOR_SPREAD: return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centers = cv2.kmeans(pixels, DOMINANT_K, None, criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS)
        counts   = np.bincount(labels.flatten())
        return np.clip(centers[int(np.argmax(counts))], 0, 255).astype(np.uint8)

    def _clusters_are_distinguishable(self, centers: np.ndarray, order: np.ndarray) -> bool:
        c0, c1 = centers[order[0]].astype(float), centers[order[1]].astype(float)
        return float(np.linalg.norm(c0 - c1)) >= _MIN_CLUSTER_DISTANCE

    def _calibrate(self) -> None:
        role_names = ["team_a", "team_b", "goalkeeper"]
        if len(self._color_buffer) < TEAM_K:
            self._use_fallback_colors("Not enough colorful crops")
            return

        data     = np.array(self._color_buffer, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
        _, labels, centers = cv2.kmeans(data, TEAM_K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        labels = labels.flatten()
        counts = np.bincount(labels, minlength=TEAM_K)
        order  = np.argsort(-counts)

        if not self._clusters_are_distinguishable(centers, order):
            self._use_fallback_colors("Clusters too similar")
            return

        self.profiles = []
        for rank, cluster_idx in enumerate(order):
            center_lab = np.clip(centers[cluster_idx], 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(center_lab.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
            self.profiles.append(TeamProfile(
                label=role_names[rank], center_lab=center_lab,
                color_bgr=(int(bgr[0]), int(bgr[1]), int(bgr[2])), count=int(counts[cluster_idx])
            ))
        self.is_calibrated, self.calibration_ok = True, True

    def _use_fallback_colors(self, reason: str) -> None:
        role_names = ["team_a", "team_b", "goalkeeper"]
        self.profiles = []
        for role in role_names:
            bgr_tuple = _FALLBACK_COLORS[role]
            bgr_arr   = np.array([[list(bgr_tuple)]], dtype=np.uint8)
            lab_arr   = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB)[0, 0]
            self.profiles.append(TeamProfile(label=role, center_lab=lab_arr, color_bgr=bgr_tuple, count=0))
        self.is_calibrated, self.calibration_ok = True, False