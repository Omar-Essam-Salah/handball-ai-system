"""
goalkeeper_analyzer.py — Per-shot GK analysis with JSON logging.

IHF goal geometry (metres)
──────────────────────────
  Left goal:  x = 0,   y ∈ [8.5, 11.5],  height = 2 m
  Right goal: x = 40,  y ∈ [8.5, 11.5],  height = 2 m

Shot placement 3×3 grid
───────────────────────
  Columns (y-axis of court):
    Left   y ∈ [8.50, 9.50]
    Centre y ∈ [9.50, 10.50]
    Right  y ∈ [10.50, 11.50]

  Rows (estimated from ball pixel height relative to goal ROI):
    Top    upper  third of goal ROI
    Mid    middle third
    Bot    lower  third

  Cell labels: TL TC TR  ML MC MR  BL BC BR

State machine per shot
───────────────────────
  IDLE → APPROACHING (ball velocity > threshold, vector points toward goal)
       → CONTACT     (ball speed drops sharply OR trajectory reverses)
       → LOGGED      (write JSON entry, reset)

Dependencies
────────────
  • pose_analyzer.Keypoints  — for GK and shooter wrist positions
  • court_detector / court_mapper — CourtDetector.pixel_to_meter()

Usage
─────
  analyzer = GoalkeeperAnalyzer(fps=25, log_path="logs/gk_log.json")
  analyzer.set_court(court_detector)           # for metre transforms
  analyzer.set_gk_track_id(gk_id, "left")     # or "right"

  # each frame:
  result = analyzer.update(frame_n, tracks, ball_track, pose_results)
  # result is None or a ShotResult when a shot is resolved
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# COCO keypoint indices (from pose_analyzer)
L_WRIST, R_WRIST = 9, 10
L_HIP,   R_HIP   = 11, 12
L_ANKLE, R_ANKLE = 15, 16
NOSE             = 0


# ── IHF goal constants ────────────────────────────────────────────────────────

_GOAL_Y_LOW  = 8.5
_GOAL_Y_HIGH = 11.5
_GOAL_Y_MID  = 10.0
_GOAL_HEIGHT_M = 2.0   # IHF goal height (not tracked in homography y-axis)

# Pixel ROI around each goal — expanded for approach detection
_GOAL_APPROACH_M = 9.0   # detect shots within 9m of goal line


# ── Shot placement grid ───────────────────────────────────────────────────────

_COL_EDGES = (_GOAL_Y_LOW, 9.5, 10.5, _GOAL_Y_HIGH)
_COL_NAMES = ("L", "C", "R")
_ROW_NAMES = ("T", "M", "B")


def _col_from_y(y_m: float) -> str:
    for i in range(3):
        if _COL_EDGES[i] <= y_m < _COL_EDGES[i + 1]:
            return _COL_NAMES[i]
    return "C"


def _row_from_pixel(ball_y_px: float,
                    goal_top_px: float, goal_bot_px: float) -> str:
    if goal_bot_px <= goal_top_px:
        return "M"
    rel = (ball_y_px - goal_top_px) / (goal_bot_px - goal_top_px)
    rel = max(0.0, min(1.0, rel))
    if rel < 0.33:
        return "T"
    if rel < 0.67:
        return "M"
    return "B"


def grid_cell(y_m: float, ball_y_px: float,
              goal_top_px: float, goal_bot_px: float) -> str:
    return _row_from_pixel(ball_y_px, goal_top_px, goal_bot_px) + _col_from_y(y_m)


# ── Data structures ───────────────────────────────────────────────────────────

class ShotState(Enum):
    IDLE        = auto()
    APPROACHING = auto()
    CONTACT     = auto()
    LOGGED      = auto()


@dataclass
class BallSnapshot:
    frame_n: int
    cx_px:   float
    cy_px:   float
    x_m:     Optional[float]
    y_m:     Optional[float]
    speed_px: float = 0.0   # pixels/frame


@dataclass
class ShotResult:
    shot_id:            str
    frame_n:            int
    timestamp_s:        float
    shooter_track_id:   Optional[int]
    goalkeeper_track_id: Optional[int]
    goal_side:          str               # "left" | "right"
    ball_x_m:           Optional[float]
    ball_y_m:           Optional[float]
    grid_cell:          str               # e.g. "TC"
    grid_col:           str               # "L" | "C" | "R"
    grid_row:           str               # "T" | "M" | "B"
    outcome:            str               # "goal" | "save" | "missed"
    reaction_time_ms:   Optional[float]
    ball_speed_px:      float             # px/frame at shot moment
    save_rate:          float             # cumulative

    def to_dict(self) -> dict:
        return {
            "shot_id":             self.shot_id,
            "frame":               self.frame_n,
            "timestamp_s":         round(self.timestamp_s, 2),
            "shooter_track_id":    self.shooter_track_id,
            "goalkeeper_track_id": self.goalkeeper_track_id,
            "goal_side":           self.goal_side,
            "ball_position_m":     {"x": self.ball_x_m, "y": self.ball_y_m},
            "grid_cell":           self.grid_cell,
            "grid_col":            self.grid_col,
            "grid_row":            self.grid_row,
            "outcome":             self.outcome,
            "reaction_time_ms":    self.reaction_time_ms,
            "ball_speed_px_per_frame": round(self.ball_speed_px, 1),
            "save_rate_cumulative": round(self.save_rate, 4),
        }


# ── Goalkeeper ROI ─────────────────────────────────────────────────────────────

@dataclass
class GoalROI:
    side:       str                          # "left" | "right"
    goal_x_m:   float                        # 0.0 or 40.0
    top_px:     Optional[tuple[int, int]]    # pixel coords of top post
    bot_px:     Optional[tuple[int, int]]    # pixel coords of bottom post
    top_y_px:   float = 0.0                  # y-pixel of crossbar (approx)
    bot_y_px:   float = 720.0               # y-pixel of ground level at goal


# ── Main analyzer ─────────────────────────────────────────────────────────────

class GoalkeeperAnalyzer:
    SHOT_SPEED_THRESHOLD   = 18.0    # px/frame — minimum ball speed to detect shot
    CONTACT_SPEED_DROP     = 0.45    # speed drops to < 45% of peak → contact
    CONTACT_ANGLE_CHANGE   = 70.0    # degrees — trajectory reversal → contact
    REACTION_MOVE_THRESH   = 12.0    # px — GK wrist/CoM movement to count as reaction
    MAX_SHOT_FRAMES        = 60      # frames before a shot is force-resolved

    def __init__(self, fps: float = 25.0, log_path: str = "logs/gk_log.json"):
        self._fps       = fps
        self._log_path  = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._court     = None    # CourtDetector (optional)
        self._gk_tid:   Optional[int] = None
        self._goal_side: str = "left"

        # Goal ROI in pixel space
        self._goal_roi: Optional[GoalROI] = None

        # Ball tracking ring buffer
        self._ball_buf: deque[BallSnapshot] = deque(maxlen=90)

        # State machine
        self._state: ShotState = ShotState.IDLE
        self._shot_start_frame: int = 0
        self._peak_speed:       float = 0.0
        self._shooter_tid:      Optional[int] = None
        self._gk_pos_at_release: Optional[tuple[float, float]] = None
        self._gk_first_move_frame: Optional[int] = None

        # Cumulative stats
        self._total_shots = 0
        self._total_saves = 0
        self._all_shots:  list[dict] = self._load_log()

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_court(self, court_detector) -> None:
        self._court = court_detector
        self._rebuild_goal_roi()

    def set_gk_track_id(self, track_id: int, side: str = "left") -> None:
        self._gk_tid  = track_id
        self._goal_side = side
        self._rebuild_goal_roi()

    def _rebuild_goal_roi(self) -> None:
        goal_x = 0.0 if self._goal_side == "left" else 40.0
        top_px = bot_px = None

        if self._court and self._court.is_calibrated:
            # Project goal posts to pixel space
            top_px = self._court.meter_to_pixel(goal_x, _GOAL_Y_HIGH)
            bot_px = self._court.meter_to_pixel(goal_x, _GOAL_Y_LOW)

        self._goal_roi = GoalROI(
            side=self._goal_side, goal_x_m=goal_x,
            top_px=top_px, bot_px=bot_px,
            top_y_px=float(top_px[1]) if top_px else 150.0,
            bot_y_px=float(bot_px[1]) if bot_px else 650.0,
        )

    # ── Frame update ──────────────────────────────────────────────────────────

    def update(
        self,
        frame_n:      int,
        tracks:       list,
        ball_track,                 # Track with is_ball=True, or None
        pose_results: list | None = None,
    ) -> Optional[ShotResult]:
        if ball_track is None:
            return None

        snap = self._snapshot(frame_n, ball_track)
        self._ball_buf.append(snap)

        gk_pose = self._find_gk_pose(tracks, pose_results)

        if self._state == ShotState.IDLE:
            return self._check_shot_start(frame_n, snap, tracks, gk_pose)

        if self._state == ShotState.APPROACHING:
            return self._check_contact(frame_n, snap, gk_pose)

        return None

    # ── State: IDLE → APPROACHING ─────────────────────────────────────────────

    def _check_shot_start(
        self,
        frame_n:  int,
        snap:     BallSnapshot,
        tracks:   list,
        gk_pose,
    ) -> Optional[ShotResult]:
        if len(self._ball_buf) < 4:
            return None

        if snap.speed_px < self.SHOT_SPEED_THRESHOLD:
            return None

        if not self._pointing_at_goal(snap):
            return None

        self._state            = ShotState.APPROACHING
        self._shot_start_frame = frame_n
        self._peak_speed       = snap.speed_px
        self._shooter_tid      = self._find_nearest_non_gk(snap, tracks)
        self._gk_first_move_frame = None

        if gk_pose is not None:
            gk_com = self._gk_com(gk_pose)
            self._gk_pos_at_release = gk_com
        else:
            self._gk_pos_at_release = None

        return None

    # ── State: APPROACHING → CONTACT / LOGGED ─────────────────────────────────

    def _check_contact(
        self,
        frame_n: int,
        snap:    BallSnapshot,
        gk_pose,
    ) -> Optional[ShotResult]:
        self._peak_speed = max(self._peak_speed, snap.speed_px)

        # Track GK first significant movement
        if self._gk_first_move_frame is None and gk_pose is not None:
            gk_com = self._gk_com(gk_pose)
            if gk_com and self._gk_pos_at_release:
                moved = math.hypot(gk_com[0] - self._gk_pos_at_release[0],
                                   gk_com[1] - self._gk_pos_at_release[1])
                if moved >= self.REACTION_MOVE_THRESH:
                    self._gk_first_move_frame = frame_n

        contact_detected = self._detect_contact(snap)
        timeout          = (frame_n - self._shot_start_frame) > self.MAX_SHOT_FRAMES

        if not contact_detected and not timeout:
            return None

        # Resolve shot
        return self._resolve_shot(frame_n, snap)

    # ── Shot resolution ───────────────────────────────────────────────────────

    def _resolve_shot(self, frame_n: int, snap: BallSnapshot) -> ShotResult:
        outcome = self._determine_outcome(snap)
        cell, col, row = self._grid_placement(snap)
        reaction_ms    = self._compute_reaction_ms(frame_n)

        self._total_shots += 1
        if outcome == "save":
            self._total_saves += 1
        save_rate = self._total_saves / self._total_shots

        result = ShotResult(
            shot_id=str(uuid.uuid4())[:8],
            frame_n=frame_n,
            timestamp_s=frame_n / self._fps,
            shooter_track_id=self._shooter_tid,
            goalkeeper_track_id=self._gk_tid,
            goal_side=self._goal_side,
            ball_x_m=snap.x_m,
            ball_y_m=snap.y_m,
            grid_cell=cell,
            grid_col=col,
            grid_row=row,
            outcome=outcome,
            reaction_time_ms=reaction_ms,
            ball_speed_px=self._peak_speed,
            save_rate=save_rate,
        )

        self._log(result)
        self._state = ShotState.IDLE
        self._peak_speed = 0.0
        return result

    # ── Outcome determination ─────────────────────────────────────────────────

    def _determine_outcome(self, snap: BallSnapshot) -> str:
        if snap.x_m is None or snap.y_m is None:
            return "unknown"
        inside_y = _GOAL_Y_LOW <= snap.y_m <= _GOAL_Y_HIGH
        at_goal_x = (
            (self._goal_side == "left"  and snap.x_m <= 0.5) or
            (self._goal_side == "right" and snap.x_m >= 39.5)
        )
        if at_goal_x and inside_y:
            if self._detect_contact(snap):
                return "save"
            return "goal"
        return "missed"

    # ── Grid placement ────────────────────────────────────────────────────────

    def _grid_placement(self, snap: BallSnapshot) -> tuple[str, str, str]:
        y_m = snap.y_m if snap.y_m is not None else _GOAL_Y_MID
        roi = self._goal_roi

        top_y = roi.top_y_px if roi else 150.0
        bot_y = roi.bot_y_px if roi else 650.0

        row = _row_from_pixel(snap.cy_px, top_y, bot_y)
        col = _col_from_y(y_m)
        return row + col, col, row

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _snapshot(self, frame_n: int, ball_track) -> BallSnapshot:
        cx = (ball_track.x1 + ball_track.x2) / 2
        cy = (ball_track.y1 + ball_track.y2) / 2
        x_m = y_m = None
        if self._court and self._court.is_calibrated:
            r = self._court.pixel_to_meter(cx, cy)
            if r:
                x_m, y_m = r

        speed = 0.0
        if self._ball_buf:
            prev = self._ball_buf[-1]
            speed = math.hypot(cx - prev.cx_px, cy - prev.cy_px)

        return BallSnapshot(frame_n=frame_n, cx_px=cx, cy_px=cy,
                            x_m=x_m, y_m=y_m, speed_px=speed)

    def _pointing_at_goal(self, snap: BallSnapshot) -> bool:
        if len(self._ball_buf) < 3:
            return False
        prev = self._ball_buf[-3]
        dx   = snap.cx_px - prev.cx_px
        dy   = snap.cy_px - prev.cy_px

        # Check if movement is generally toward the goal side
        if self._goal_side == "left" and dx >= 0:
            return False
        if self._goal_side == "right" and dx <= 0:
            return False

        # Check proximity if metre coords available
        if snap.x_m is not None:
            goal_x = self._goal_roi.goal_x_m if self._goal_roi else 0.0
            if abs(snap.x_m - goal_x) > _GOAL_APPROACH_M:
                return False

        return True

    def _detect_contact(self, snap: BallSnapshot) -> bool:
        if self._peak_speed < 1.0:
            return False
        speed_drop = snap.speed_px < self._peak_speed * self.CONTACT_SPEED_DROP

        angle_change = False
        if len(self._ball_buf) >= 5:
            early = self._ball_buf[-5]
            mid   = self._ball_buf[-3]
            v1    = (mid.cx_px - early.cx_px, mid.cy_px - early.cy_px)
            v2    = (snap.cx_px - mid.cx_px,  snap.cy_px - mid.cy_px)
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 > 1 and n2 > 1:
                cos_a = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1 * n2)
                angle  = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
                angle_change = angle > self.CONTACT_ANGLE_CHANGE

        return speed_drop or angle_change

    def _find_gk_pose(self, tracks: list, pose_results: list | None):
        if pose_results is None or self._gk_tid is None:
            return None
        for pr in pose_results:
            if pr.track_id == self._gk_tid:
                return pr
        return None

    def _gk_com(self, gk_pose) -> Optional[tuple[float, float]]:
        kp = gk_pose.keypoints
        xs, ys = [], []
        for idx in (L_HIP, R_HIP, L_ANKLE, R_ANKLE, NOSE):
            if kp.visible(idx):
                xs.append(float(kp.xy[idx, 0]))
                ys.append(float(kp.xy[idx, 1]))
        if not xs:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    def _find_nearest_non_gk(self, snap: BallSnapshot, tracks: list) -> Optional[int]:
        best_id, best_d = None, float("inf")
        for t in tracks:
            if t.is_ball or t.track_id == self._gk_tid:
                continue
            d = math.hypot(t.cx - snap.cx_px, t.cy - snap.cy_px)
            if d < best_d:
                best_d, best_id = d, t.track_id
        return best_id

    def _compute_reaction_ms(self, contact_frame: int) -> Optional[float]:
        if self._gk_first_move_frame is None:
            return None
        delta_frames = self._gk_first_move_frame - self._shot_start_frame
        if delta_frames < 0:
            return None
        return round(delta_frames / self._fps * 1000.0, 1)

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw goal ROI box, ball trajectory trail, and last shot grid on frame.
        Returns a copy; does not mutate the input.
        """
        out = frame.copy()
        roi = self._goal_roi

        # Goal ROI bounding box
        if roi and roi.top_px and roi.bot_px:
            cv2.rectangle(out, roi.top_px, roi.bot_px, (0, 255, 180), 2)
            # 3×3 grid overlay
            x0, x1_g = min(roi.top_px[0], roi.bot_px[0]), max(roi.top_px[0], roi.bot_px[0])
            y0, y1_g = min(roi.top_px[1], roi.bot_px[1]), max(roi.top_px[1], roi.bot_px[1])
            for frac in (1/3, 2/3):
                xm = int(x0 + (x1_g - x0) * frac)
                ym = int(y0 + (y1_g - y0) * frac)
                cv2.line(out, (xm, y0), (xm, y1_g), (0, 220, 140), 1)
                cv2.line(out, (x0, ym), (x1_g, ym), (0, 220, 140), 1)

        # Ball trail
        pts = list(self._ball_buf)[-20:]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            col   = (int(0 * alpha), int(220 * alpha), int(255 * alpha))
            cv2.line(out,
                     (int(pts[i-1].cx_px), int(pts[i-1].cy_px)),
                     (int(pts[i].cx_px),   int(pts[i].cy_px)),
                     col, 2, cv2.LINE_AA)

        # State badge
        color = {
            ShotState.IDLE:        (100, 100, 100),
            ShotState.APPROACHING: (0, 200, 255),
            ShotState.CONTACT:     (0, 255, 80),
            ShotState.LOGGED:      (255, 200, 0),
        }.get(self._state, (180, 180, 180))
        cv2.putText(out, f"GK: {self._state.name}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return out

    # ── JSON persistence ──────────────────────────────────────────────────────

    def _log(self, result: ShotResult) -> None:
        self._all_shots.append(result.to_dict())
        self._flush()

    def _flush(self) -> None:
        summary = {
            "total_shots":       self._total_shots,
            "saves":             self._total_saves,
            "goals_conceded":    self._total_shots - self._total_saves,
            "save_rate":         round(self._total_saves / max(self._total_shots, 1), 4),
            "avg_reaction_ms":   self._avg_reaction(),
            "placement_counts":  self._placement_counts(),
        }
        payload = {"shots": self._all_shots, "summary": summary}
        try:
            self._log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_log(self) -> list[dict]:
        if self._log_path.exists():
            try:
                d = json.loads(self._log_path.read_text("utf-8"))
                shots = d.get("shots", [])
                self._total_shots = len(shots)
                self._total_saves = sum(1 for s in shots if s.get("outcome") == "save")
                return shots
            except Exception:
                pass
        return []

    def _avg_reaction(self) -> Optional[float]:
        rts = [s["reaction_time_ms"] for s in self._all_shots
               if s.get("reaction_time_ms") is not None]
        return round(float(np.mean(rts)), 1) if rts else None

    def _placement_counts(self) -> dict[str, int]:
        all_cells = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]
        counts = {c: 0 for c in all_cells}
        for s in self._all_shots:
            cell = s.get("grid_cell", "")
            if cell in counts:
                counts[cell] += 1
        return counts

    # ── Public stats ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "total_shots":    self._total_shots,
            "saves":          self._total_saves,
            "goals_conceded": self._total_shots - self._total_saves,
            "save_rate":      round(self._total_saves / max(self._total_shots, 1), 4),
            "avg_reaction_ms": self._avg_reaction(),
            "placement_counts": self._placement_counts(),
            "state":          self._state.name,
            "gk_track_id":    self._gk_tid,
            "goal_side":      self._goal_side,
        }
