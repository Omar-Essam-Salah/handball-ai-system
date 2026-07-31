"""
SAHI-Backed Detector — sliced inference for distant / small players.

Why this exists
───────────────
The standard Detector resizes the input frame to 640px before YOLO inference.
On a 1920×1080 broadcast, a player at the far end of the court ends up
~10-20px tall — below YOLO's effective detection scale. They get missed
entirely or come back with conf < 0.10.

SAHI (Slicing Aided Hyper Inference) fixes this by:
    1. Splitting the input frame into overlapping 640×640 slices.
    2. Running YOLO on each slice at full resolution.
    3. Merging detections back into the original frame coordinates.

For the distant player, the slice containing them is ~640px wide instead of
1920px → the player now occupies a much larger fraction of the slice
(effective scale 5-10× larger) → YOLO catches them cleanly.

Trade-off: ~3-4× slower because we run 4-9 slices instead of 1 full frame.
Default OFF (`--use-sahi` enables it). Recommended for:
- Broadcast wide-angle shots
- Static camera with players spread across full court
- Recording for offline analysis where FPS matters less

Compatibility
─────────────
Wraps the existing `Detector` class — keeps the same `.detect(frame)` API and
returns the same `list[Detection]`. Just swaps the internal YOLO inference
loop with `sahi.predict.get_sliced_prediction`.

Usage in pipeline:
    if args.use_sahi:
        from sahi_detector import SAHIDetector
        detector = SAHIDetector(device=..., model=..., ball_conf=...)
    else:
        detector = Detector(...)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


PERSON_CLASS = 0
BALL_CLASS   = 32


@dataclass
class Detection:
    """Mirrors detector.Detection exactly so the rest of the pipeline can
    use SAHIDetector as a drop-in replacement."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    keypoints: np.ndarray | None = None

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


class SAHIDetector:
    """Drop-in replacement for Detector that uses SAHI sliced inference.

    Design notes
    ────────────
    * POSE model: SAHI doesn't natively support pose outputs. We run pose on
      the FULL frame at higher imgsz=960 (better recall on distant players
      without slicing complexity for keypoints). This is the "best of both"
      compromise.
    * BALL model: runs sliced inference at imgsz=640 per slice with 25% overlap.
      Distant balls (e.g. just released from a far-court shot) get picked up.
    * Outputs: same `list[Detection]` shape as the base Detector — pipeline
      code doesn't need to change.
    """

    def __init__(
        self,
        device:           str   = "cuda:0",
        model:            str   = "yolov8s-pose.pt",
        conf:             float = 0.25,
        ball_conf:        float = 0.02,
        imgsz:            int   = 640,
        # Pose model runs full-frame at higher imgsz for distant-player recall.
        pose_imgsz:       int   = 960,
        # SAHI ball-slicing parameters
        slice_height:     int   = 640,
        slice_width:      int   = 640,
        overlap_ratio:    float = 0.25,
        # Geometric ball filters (kept identical to base Detector)
        small_conf:       float = 0.15,
        large_conf:       float = 0.30,
        small_area_px:    int   = 1000,
        ball_min_area_px: int   = 20,
        ball_max_area_px: int   = 9000,
        ball_aspect_max:  float = 4.0,
    ):
        from ultralytics import YOLO
        from sahi import AutoDetectionModel

        self.device           = device
        self.conf             = conf
        self.ball_conf        = ball_conf
        self.imgsz            = imgsz
        self.pose_imgsz       = pose_imgsz
        self.slice_height     = int(slice_height)
        self.slice_width      = int(slice_width)
        self.overlap_ratio    = float(overlap_ratio)
        self.small_conf       = small_conf
        self.large_conf       = large_conf
        self.small_area_px    = small_area_px
        self.ball_min_area_px = ball_min_area_px
        self.ball_max_area_px = ball_max_area_px
        self.ball_aspect_max  = ball_aspect_max

        self._use_half = ("cuda" in str(device))

        # 1. Pose model — full-frame at higher imgsz (no slicing for pose)
        self.model_pose = YOLO(model, task="pose")

        # 2. Ball model — fine-tuned weights if available, fallback to COCO
        _project_root = Path(__file__).parent.parent
        _ball_weights = _project_root / "handball_ball.pt"
        if _ball_weights.exists():
            ball_model_path = str(_ball_weights)
            self._ball_classes = [0]
            print(f"[SAHIDetector] using fine-tuned ball weights: {_ball_weights.name}")
            if self.ball_conf < 0.20:
                self.ball_conf = 0.20
                print(f"[SAHIDetector] auto-bumped ball_conf to 0.20 (fine-tuned model is confident)")
        else:
            ball_model_path = "yolov8n.pt"
            self._ball_classes = [BALL_CLASS]
            print(f"[SAHIDetector] using COCO yolov8n.pt for ball")

        # SAHI wrapper around the ball model
        self.sahi_ball = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=ball_model_path,
            confidence_threshold=self.ball_conf,
            device=device,
        )

        # Warmup pose model
        dummy = np.zeros((pose_imgsz, pose_imgsz, 3), dtype=np.uint8)
        self.model_pose(dummy, device=self.device, classes=[PERSON_CLASS],
                        imgsz=pose_imgsz, verbose=False, half=self._use_half)

        print(f"[SAHIDetector] Ready on {device}  pose={model}@{pose_imgsz}  "
              f"ball=SAHI-sliced({slice_width}×{slice_height}, overlap={overlap_ratio})")

    # ────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        # ── 1. Pose Engine — full-frame at higher imgsz ──
        # Higher imgsz already helps distant-player recall significantly.
        # We don't slice the pose model because:
        #   (a) SAHI doesn't natively support pose outputs, and
        #   (b) split keypoints don't merge cleanly across slice boundaries.
        for r in self.model_pose(
            frame,
            device=self.device,
            classes=[PERSON_CLASS],
            imgsz=self.pose_imgsz,
            conf=self.small_conf,
            verbose=False,
            half=self._use_half,
        ):
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)

                thresh = self.small_conf if area < self.small_area_px else self.large_conf
                if conf_val < thresh:
                    continue

                kpts = None
                if r.keypoints is not None and r.keypoints.data.shape[1] > 0:
                    kpts = r.keypoints.data[i].cpu().numpy()

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf_val,
                    class_id=PERSON_CLASS,
                    keypoints=kpts,
                ))

        # ── 2. Ball Engine — SAHI sliced inference ──
        # The frame is split into overlapping slices, each run through YOLO
        # at full resolution, then results merged via NMS in original coords.
        try:
            from sahi.predict import get_sliced_prediction
            result = get_sliced_prediction(
                frame,
                self.sahi_ball,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                perform_standard_pred=False,   # skip the extra full-frame pass
                postprocess_type="NMS",
                postprocess_match_threshold=0.5,
                verbose=0,
            )
            for det in result.object_prediction_list:
                # SAHI's COCO ID can be 0 (single class) or 32 (COCO) — normalise
                bbox = det.bbox
                x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                conf_val = float(det.score.value)
                area = (x2 - x1) * (y2 - y1)

                if area < self.ball_min_area_px or area > self.ball_max_area_px:
                    continue

                w_det, h_det = x2 - x1, y2 - y1
                if h_det > 0:
                    ratio = w_det / h_det
                    if ratio > self.ball_aspect_max or ratio < 1.0 / self.ball_aspect_max:
                        continue

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf_val,
                    class_id=BALL_CLASS,
                    keypoints=None,
                ))
        except Exception as e:
            if not getattr(self, "_sahi_err_logged", False):
                print(f"[SAHIDetector] ball sliced inference failed: "
                      f"{type(e).__name__}: {e}")
                self._sahi_err_logged = True

        return detections
