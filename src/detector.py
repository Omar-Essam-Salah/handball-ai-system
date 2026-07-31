"""
YOLOv8 / TensorRT wrapper — Dual-Engine (Pose + Object) detection.

This is the simple, stable version (matches the old D:\\HandBall V2 project).
The lesson from earlier experiments: stacking many filters in series
(ROI cropping, colour validators, tight area + aspect bounds) compounds
false-rejection. Trust YOLO for raw recall; the tracker's scoring (in
tracker.py) sorts out the false positives via hand-magnet + anti-shoe.

Key design points
─────────────────
• Dual-Model Inference: a Pose model for players (to get wrist/ankle keypoints)
  and a lightweight Nano model for the ball (Pose models cannot see objects).
• Keypoints extraction: each person detection includes a 17-point COCO skeleton.
• Dynamic confidence: separate thresholds for small vs large persons, plus
  geometric filters for the ball (area + aspect ratio — kept very permissive).
"""

from dataclasses import dataclass

import numpy as np


PERSON_CLASS = 0      # COCO class 0
BALL_CLASS   = 32     # COCO class 32 — "sports ball"


def _resolve_weight(pt_path: str):
    """Plaintext-first weight resolution for the DRM layer.

    Returns (path, is_encrypted) or (None, False) if neither a raw .pt nor an
    encrypted .pt.enc exists. In DEV the raw .pt is present → nothing changes.
    When you SHIP (delete the .pt, keep only the .enc) the encrypted path engages
    automatically. HB_USE_ENCRYPTED_WEIGHTS=1 forces the .enc path for testing.
    """
    import os
    from pathlib import Path
    p = Path(pt_path)
    enc = Path(str(pt_path) + ".enc")
    force = os.environ.get("HB_USE_ENCRYPTED_WEIGHTS") == "1"
    if enc.exists() and (force or not p.exists()):
        return (str(enc), True)
    if p.exists():
        return (str(p), False)
    return (None, False)


def _load_yolo(path: str, task: str, is_encrypted: bool):
    """Load a YOLO model from a plaintext .pt, or decrypt an .enc in RAM under a
    valid license (delegates to licensing.secure_loader)."""
    from ultralytics import YOLO
    if not is_encrypted:
        return YOLO(path, task=task)
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from licensing.secure_loader import load_yolo as _secure_load
    print(f"[Detector] ENCRYPTED weights {_P(path).name} — RAM-decrypt, license-gated")
    return _secure_load(path, task=task)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int                          # 0 = person, 32 = ball
    keypoints: np.ndarray | None = None    # (17, 3) -> [x, y, conf] per joint

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS

    @property
    def is_ball(self) -> bool:
        return self.class_id == BALL_CLASS


class Detector:
    """Wraps YOLOv8 Pose + YOLOv8 Object detection into a unified pipeline."""

    def __init__(
        self,
        device:           str   = "cuda:0",
        model:            str   = "yolov8s-pose.pt",
        conf:             float = 0.25,
        ball_conf:        float = 0.02,    # trust the nano detector — let weak ball candidates through
        imgsz:            int   = 640,
        ball_imgsz:       int   = 960,     # ball is SMALL — run the (cheap) ball net at higher res
        small_conf:       float = 0.15,
        large_conf:       float = 0.30,
        small_area_px:    int   = 1000,
        ball_min_area_px: int   = 20,
        ball_max_area_px: int   = 9000,
        ball_aspect_max:  float = 4.0,     # fixed; covers motion-blurred ellipses without dynamic logic
    ):
        from ultralytics import YOLO

        self.device           = device
        self.conf             = conf
        self.ball_conf        = ball_conf
        self.imgsz            = imgsz
        # Ball model runs at a higher resolution than the pose model: the ball
        # is only ~10-20px in 1080p wide shots, and at imgsz=640 YOLO misses it
        # ~half the time. Bumping ONLY the lightweight nano ball net to 960
        # lifts recall ~49%→58% while the expensive pose net stays at 640, so
        # players (which are large) keep their speed. Measured on handball.mp4.
        self.ball_imgsz       = max(imgsz, int(ball_imgsz))
        self.small_conf       = small_conf
        self.large_conf       = large_conf
        self.small_area_px    = small_area_px
        self.ball_min_area_px = ball_min_area_px
        self.ball_max_area_px = ball_max_area_px
        self.ball_aspect_max  = ball_aspect_max

        self._use_half = ("cuda" in str(device))

        # 1. Pose model (players) — plaintext-first, encrypted-fallback (DRM).
        _pose_resolved, _pose_enc = _resolve_weight(model)
        self.model_pose = _load_yolo(_pose_resolved or model, "pose", _pose_enc)

        # 2. Ball model — prefer the fine-tuned handball-specific weights when
        # they exist (recall ~80% vs COCO's ~30%), fall back to yolov8n. Also
        # honours encrypted (.enc) weights via the same DRM-aware loader.
        from pathlib import Path as _Path
        _project_root = _Path(__file__).parent.parent
        _ball_resolved, _ball_enc = _resolve_weight(str(_project_root / "handball_ball.pt"))
        if _ball_resolved is not None:
            self.model_ball = _load_yolo(_ball_resolved, "detect", _ball_enc)
            self._ball_classes = [0]      # fine-tuned model uses class 0 = ball
            # We DON'T auto-raise ball_conf any more. After several training
            # cycles we observed that user-corrected fine-tunes have median
            # conf ~0.15-0.20 (much lower than the original synthetic
            # auto-label model at 0.72). Raising the floor to 0.20 dropped
            # ~22% of real-ball frames. Trust the CLI value the user passed,
            # but enforce a sane minimum of 0.04.
            if self.ball_conf < 0.02:
                print(f"[Detector] floor ball_conf {self.ball_conf} → 0.02")
                self.ball_conf = 0.02
            print(f"[Detector] using fine-tuned ball weights: {_Path(_ball_resolved).name}  "
                  f"(ball_conf={self.ball_conf:.2f})")
        else:
            self.model_ball = _load_yolo("yolov8n.pt", "detect", False)
            self._ball_classes = [BALL_CLASS]  # COCO model uses class 32
            print(f"[Detector] using COCO yolov8n.pt for ball (consider running training/auto_label.py + train_ball.py)")

        # Warmup
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model_pose(dummy, device=self.device, classes=[PERSON_CLASS],
                         imgsz=imgsz, verbose=False, half=self._use_half)
        ball_dummy = np.zeros((self.ball_imgsz, self.ball_imgsz, 3), dtype=np.uint8)
        self.model_ball(ball_dummy, device=self.device, classes=self._ball_classes,
                         imgsz=self.ball_imgsz, verbose=False, half=self._use_half)

        print(f"[Detector] Dual-Engine Ready on {device} "
              f"(Pose: {model}@{imgsz} | Ball: yolov8n.pt@{self.ball_imgsz} | "
              f"area=[{ball_min_area_px}, {ball_max_area_px}] | "
              f"aspect_max={ball_aspect_max})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Players (pose) + ball, full frame. Kept for the standalone pipeline."""
        return self.detect_players(frame) + self.detect_ball(frame)

    # ──────────────────────────────────────────────────────────────────
    def detect_players(self, frame: np.ndarray) -> list[Detection]:
        """Pose engine only (players + 17 keypoints). The expensive model — the
        pipeline runs this every ``infer_every`` frames."""
        detections: list[Detection] = []
        for r in self.model_pose(
            frame,
            device=self.device,
            classes=[PERSON_CLASS],
            imgsz=self.imgsz,
            conf=self.small_conf,
            verbose=False,
            half=self._use_half,
        ):
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                area     = (x2 - x1) * (y2 - y1)
                thresh = self.small_conf if area < self.small_area_px else self.large_conf
                if conf_val < thresh:
                    continue
                kpts = None
                if r.keypoints is not None and r.keypoints.data.shape[1] > 0:
                    kpts = r.keypoints.data[i].cpu().numpy()
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf_val,
                    class_id=PERSON_CLASS, keypoints=kpts))
        return detections

    # ──────────────────────────────────────────────────────────────────
    def _ball_from(self, results, ox: float = 0.0, oy: float = 0.0) -> list[Detection]:
        out: list[Detection] = []
        amax = self.ball_aspect_max
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w_det, h_det = x2 - x1, y2 - y1
                area = w_det * h_det
                if area < self.ball_min_area_px or area > self.ball_max_area_px:
                    continue
                if h_det > 0:
                    ratio = w_det / h_det
                    if ratio > amax or ratio < 1.0 / amax:
                        continue
                out.append(Detection(
                    x1=x1 + ox, y1=y1 + oy, x2=x2 + ox, y2=y2 + oy,
                    confidence=float(box.conf[0]), class_id=BALL_CLASS, keypoints=None))
        return out

    def detect_ball(self, frame: np.ndarray,
                    roi_center: tuple | None = None,
                    roi_size: int = 384) -> list[Detection]:
        """Ball engine — cheap, so the pipeline runs it EVERY frame.

        Runs a full-frame pass, plus (when a recent ball position is known) a
        high-resolution pass on a crop around it: a tiny far-away ball becomes
        large inside the crop, which is what recovers the detections the
        downscaled full frame loses. This is the main fix for small balls on
        high-resolution / wide-angle footage."""
        dets = self._ball_from(self.model_ball(
            frame, device=self.device, classes=self._ball_classes,
            imgsz=self.ball_imgsz, conf=self.ball_conf, verbose=False,
            half=self._use_half))

        if roi_center is not None:
            H, W = frame.shape[:2]
            cx, cy = roi_center
            x1 = max(0, int(cx - roi_size / 2)); y1 = max(0, int(cy - roi_size / 2))
            x2 = min(W, x1 + roi_size);          y2 = min(H, y1 + roi_size)
            if x2 - x1 >= 32 and y2 - y1 >= 32:
                crop = frame[y1:y2, x1:x2]
                dets += self._ball_from(self.model_ball(
                    crop, device=self.device, classes=self._ball_classes,
                    imgsz=640, conf=self.ball_conf, verbose=False,
                    half=self._use_half), ox=x1, oy=y1)
        return dets
