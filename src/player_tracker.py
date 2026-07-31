"""
PlayerTracker — BoT-SORT wrapper tuned for OBLIQUE/STAND-MOUNTED CAMERA views.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

PERSON_CLASS = 0
BALL_CLASS   = 32

@dataclass
class Track:
    track_id:   int
    x1:         float
    y1:         float
    x2:         float
    y2:         float
    confidence: float
    class_id:   int

    @property
    def cx(self) -> float: return (self.x1 + self.x2) * 0.5
    @property
    def cy(self) -> float: return (self.y1 + self.y2) * 0.5
    @property
    def width(self) -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def is_ball(self) -> bool: return self.class_id == BALL_CLASS

    def foot_xy(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, self.y2)

class PlayerTracker:
    REID_MODEL = Path("osnet_ain_x1_0_msmt17.pt")
    BALL_TRACK_ID = 9999

    def __init__(
        self,
        device:        str   = "cuda:0",
        frame_rate:    int   = 30,
        track_buffer:  int   = 150,    
        match_thresh:  float = 0.95,
        track_high_thresh:  float = 0.25,
        track_low_thresh:   float = 0.05,
        new_track_thresh:   float = 0.70,
        proximity_thresh:   float = 0.50,
        appearance_thresh:  float = 0.10,
        with_reid:     bool  = True,
        # الإلغاء الجذري لحسابات حركة الكاميرا الوهمية التي تدمر التتبع
        cmc_method:    str   = "none", 
    ):
        import torch
        try:
            from boxmot import BotSort
        except ImportError:
            from boxmot.trackers import BotSort

        dev  = torch.device(device)
        half = (dev.type != "cpu")

        kwargs = dict(
            reid_weights      = self.REID_MODEL,
            device            = dev,
            half              = half,
            cmc_method        = cmc_method,
            frame_rate        = frame_rate,
            track_buffer      = track_buffer,
            match_thresh      = match_thresh,
            track_high_thresh = track_high_thresh,
            track_low_thresh  = track_low_thresh,
            new_track_thresh  = new_track_thresh,
            proximity_thresh  = proximity_thresh,
            appearance_thresh = appearance_thresh,
            with_reid         = with_reid,
        )

        self._tracker = self._init_with_fallback(BotSort, kwargs)

    @staticmethod
    def _init_with_fallback(cls, kwargs: dict):
        attempt = dict(kwargs)
        unsupported: list[str] = []
        while True:
            try:
                return cls(**attempt)
            except TypeError as exc:
                msg = str(exc)
                bad: Optional[str] = None
                for k in list(attempt.keys()):
                    if f"'{k}'" in msg:
                        bad = k; break
                if bad is None: raise
                attempt.pop(bad)
                unsupported.append(bad)
                if not attempt or len(unsupported) > 12: raise

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[Track]:
        det_arr = self._coerce_detections(detections)

        if det_arr.size:
            person_mask = det_arr[:, 5].astype(int) == PERSON_CLASS
            ball_mask   = det_arr[:, 5].astype(int) == BALL_CLASS
        else:
            person_mask = np.zeros(0, dtype=bool)
            ball_mask   = np.zeros(0, dtype=bool)

        person_dets = det_arr[person_mask] if det_arr.size else np.empty((0, 6), dtype=np.float32)
        ball_dets   = det_arr[ball_mask]   if det_arr.size else np.empty((0, 6), dtype=np.float32)

        try:
            raw = self._tracker.update(person_dets.astype(np.float32), frame)
        except Exception as exc:
            raw = []

        tracks: list[Track] = []
        for t in raw:
            tracks.append(Track(
                x1=float(t[0]), y1=float(t[1]), x2=float(t[2]), y2=float(t[3]),
                track_id=int(t[4]), confidence=float(t[5]), class_id=int(t[6]),
            ))

        if ball_dets.shape[0]:
            best = ball_dets[ball_dets[:, 4].argmax()]
            tracks.append(Track(
                x1=float(best[0]), y1=float(best[1]), x2=float(best[2]), y2=float(best[3]),
                track_id=self.BALL_TRACK_ID, confidence=float(best[4]), class_id=BALL_CLASS,
            ))
        return tracks

    @staticmethod
    def _coerce_detections(detections) -> np.ndarray:
        if isinstance(detections, np.ndarray):
            return detections.astype(np.float32) if detections.size else np.empty((0, 6), dtype=np.float32)
        rows = []
        for d in detections:
            rows.append([
                float(d.x1), float(d.y1), float(d.x2), float(d.y2),
                float(getattr(d, "confidence", getattr(d, "conf", 0.0))),
                float(getattr(d, "class_id", getattr(d, "cls", 0))),
            ])
        return np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)