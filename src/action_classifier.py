"""
action_classifier.py — Pass / Dribble / Shot / Move classification from
tracking data (ball trajectory + player positions).

Classification model (all speeds in px/frame at the processing resolution,
convertible to m/s when a homography is available):

    v(t)      = |ball(t) − ball(t−1)|                     ball speed
    dir(t)    = unit(ball(t) − ball(t−1))                 ball direction
    goal_cone = angle(dir, goal_center − ball) < θ_goal    aimed at goal?

  PASS   (hand → hand):
      1. ball exits possessor A's radius (R_pos) with speed v ≥ V_PASS_MIN
      2. within T_PASS_MAX frames it enters player B's radius (B ≠ A, any team;
         same team ⇒ completed pass, other team ⇒ interception)
      3. NO periodic vertical oscillation while in flight (that's a dribble)
      4. flight path is ballistic: direction change between consecutive
         steps < θ_straight  (a rolling deflected ball zig-zags)

  DRIBBLE (locomotion while bouncing):
      1. ball stays within R_dribble of the SAME player for ≥ N_DRB frames
      2. vertical position y(t) oscillates:  ≥ 2 sign flips of dy/dt within
         the window AND peak-to-peak amplitude ≥ A_MIN px
      3. the player is moving:  |player(t) − player(t−N)| ≥ D_MOVE px

  SHOT (high-velocity attempt at goal):
      1. ball exits possessor's radius with v ≥ V_SHOT_MIN  (≫ pass speed)
      2. direction within θ_goal of either goal centre
      3. sustained for ≥ 2 consecutive frames (kills single-frame jitter)

  MOVE (off-ball run): any player displacement ≥ D_MOVE px within the window
      while NOT the possessor — rendered with solid arrows.

The classifier is deliberately stateless per segment: feed it per-frame
observations via update(); it emits ActionSegment objects when a segment
closes.  It does NOT replace HandballEngine's event logic — it produces the
*visual* classification used by the tactical renderer (ihf_notation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Tunables (processing resolution: 1280 px wide frame) ────────────────────
R_POSSESS_PX   = 110.0    # ball within this radius of a player = possessed
R_DRIBBLE_PX   = 130.0    # dribble keeps the ball a bit farther (bounce)
V_PASS_MIN     = 6.0      # px/frame — slower than this is a fumble/roll
V_SHOT_MIN     = 18.0     # px/frame — matches handball_rules.SHOT_SPEED_MIN_PX
T_PASS_MAX     = 30       # frames the ball may fly before reaching a receiver
N_DRIBBLE_MIN  = 8        # frames of sustained bouncing to call a dribble
A_BOUNCE_MIN   = 9.0      # px peak-to-peak vertical amplitude of a bounce
D_MOVE_MIN     = 40.0     # px displacement to draw a movement arrow
THETA_GOAL_DEG = 22.0     # shot must aim within this cone of a goal centre
THETA_STRAIGHT = 35.0     # max per-step direction change for a ballistic pass


@dataclass
class ActionSegment:
    """One classified action, ready for the renderer."""
    kind:        str                  # "pass" | "dribble" | "shot" | "move"
    start_frame: int
    end_frame:   int
    from_tid:    Optional[int]       # actor (passer / dribbler / shooter)
    to_tid:      Optional[int]       # receiver (passes only)
    team:        Optional[str]
    path:        list = field(default_factory=list)   # [(x, y)] ball or player
    meta:        dict = field(default_factory=dict)

    @property
    def p_from(self):
        return self.path[0] if self.path else None

    @property
    def p_to(self):
        return self.path[-1] if self.path else None


class ActionClassifier:
    """Feed per-frame observations; collect classified ActionSegments."""

    def __init__(self, frame_w: int, frame_h: int, fps: float = 25.0):
        self.fw, self.fh, self.fps = frame_w, frame_h, fps
        self.goals = (np.array([0.0, frame_h * 0.5]),          # left goal centre
                      np.array([float(frame_w), frame_h * 0.5]))  # right goal centre
        self._ball_hist:   list[tuple[int, float, float]] = []  # (fn, x, y)
        self._possessor:   Optional[int] = None
        self._possessor_team: Optional[str] = None
        self._flight:      Optional[dict] = None   # in-flight ball segment
        self._dribble:     Optional[dict] = None
        self._player_hist: dict[int, list] = {}    # tid → [(fn, x, y)]
        self.segments:     list[ActionSegment] = []

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _speed(a, b) -> float:
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))

    def _aimed_at_goal(self, pos, direction) -> bool:
        d = np.asarray(direction, np.float64)
        n = np.linalg.norm(d)
        if n < 1e-6:
            return False
        d /= n
        for g in self.goals:
            to_goal = g - np.asarray(pos, np.float64)
            tn = np.linalg.norm(to_goal)
            if tn < 1e-6:
                continue
            cosang = float(np.dot(d, to_goal / tn))
            if cosang > np.cos(np.radians(THETA_GOAL_DEG)):
                return True
        return False

    @staticmethod
    def _nearest_player(tracks: list[dict], xy) -> tuple[Optional[dict], float]:
        best, best_d = None, 1e9
        for t in tracks:
            d = float(np.hypot(t["cx"] - xy[0], t["cy"] - xy[1]))
            if d < best_d:
                best, best_d = t, d
        return best, best_d

    # ── main update ──────────────────────────────────────────────────────────
    def update(self, frame_n: int, tracks: list[dict],
               ball_xy: Optional[tuple]) -> list[ActionSegment]:
        """`tracks`: [{tid, cx, cy, team, ...}] player detections this frame.
        Returns segments that CLOSED this frame (may be empty)."""
        closed: list[ActionSegment] = []

        for t in tracks:
            self._player_hist.setdefault(t["tid"], []).append(
                (frame_n, float(t["cx"]), float(t["cy"])))
            if len(self._player_hist[t["tid"]]) > 90:
                self._player_hist[t["tid"]] = self._player_hist[t["tid"]][-90:]

        if ball_xy is None:
            return closed
        self._ball_hist.append((frame_n, float(ball_xy[0]), float(ball_xy[1])))
        if len(self._ball_hist) > 120:
            self._ball_hist = self._ball_hist[-120:]

        holder, dist = self._nearest_player(tracks, ball_xy)

        # ---- in-flight segment resolution (pass or shot in progress) -------
        if self._flight is not None:
            fl = self._flight
            fl["path"].append(tuple(ball_xy))
            v = self._speed(fl["path"][-2], fl["path"][-1]) if len(fl["path"]) > 1 else 0.0

            # direction-change sanity for ballistic flight
            if len(fl["path"]) >= 3:
                a, b, c = (np.array(p) for p in fl["path"][-3:])
                u1, u2 = b - a, c - b
                if (np.linalg.norm(u1) > 1e-3 and np.linalg.norm(u2) > 1e-3):
                    cosang = float(np.dot(u1, u2) /
                                   (np.linalg.norm(u1) * np.linalg.norm(u2)))
                    if cosang < np.cos(np.radians(THETA_STRAIGHT)):
                        fl["ballistic"] = False

            reached = (holder is not None and dist <= R_POSSESS_PX
                       and holder["tid"] != fl["from_tid"])
            expired = (frame_n - fl["start_frame"]) > T_PASS_MAX or v < 1.0

            if fl["kind"] == "shot" and (expired or reached):
                closed.append(ActionSegment(
                    "shot", fl["start_frame"], frame_n, fl["from_tid"], None,
                    fl["team"], fl["path"],
                    {"peak_speed": fl["peak_speed"]}))
                self._flight = None
            elif reached:
                same_team = (holder.get("team") == fl["team"])
                closed.append(ActionSegment(
                    "pass", fl["start_frame"], frame_n,
                    fl["from_tid"], holder["tid"], fl["team"], fl["path"],
                    {"completed": same_team,
                     "intercepted": not same_team,
                     "ballistic": fl.get("ballistic", True)}))
                self._flight = None
                self._possessor = holder["tid"]
                self._possessor_team = holder.get("team")
            elif expired:
                self._flight = None            # lost/rolling ball — no segment

        # ---- possession & launch detection ---------------------------------
        if self._flight is None:
            if holder is not None and dist <= R_POSSESS_PX:
                # ball is with a player — maybe a dribble in progress
                self._update_dribble(frame_n, holder, ball_xy, closed)
                self._possessor = holder["tid"]
                self._possessor_team = holder.get("team")
            elif (self._possessor is not None
                  and len(self._ball_hist) >= 2):
                # ball just left the possessor's radius → classify the launch
                v = self._speed(self._ball_hist[-2][1:], self._ball_hist[-1][1:])
                direction = (self._ball_hist[-1][1] - self._ball_hist[-2][1],
                             self._ball_hist[-1][2] - self._ball_hist[-2][2])
                if v >= V_SHOT_MIN and self._aimed_at_goal(ball_xy, direction):
                    kind = "shot"
                elif v >= V_PASS_MIN:
                    kind = "pass"
                else:
                    kind = None
                if kind:
                    self._close_dribble(frame_n, closed)
                    self._flight = {
                        "kind": kind, "from_tid": self._possessor,
                        "team": self._possessor_team,
                        "start_frame": frame_n,
                        "path": [tuple(ball_xy)],
                        "peak_speed": v, "ballistic": True,
                    }
                self._possessor = None

        # ---- off-ball movement arrows (sampled, not per-frame) -------------
        if frame_n % 25 == 0:
            closed.extend(self._movement_segments(frame_n))

        self.segments.extend(closed)
        return closed

    # ── dribble state machine ────────────────────────────────────────────────
    def _update_dribble(self, frame_n, holder, ball_xy, closed):
        d = self._dribble
        if d is None or d["tid"] != holder["tid"]:
            self._close_dribble(frame_n, closed)
            self._dribble = {"tid": holder["tid"], "team": holder.get("team"),
                             "start_frame": frame_n,
                             "ball_y": [float(ball_xy[1])],
                             "player_path": [(holder["cx"], holder["cy"])]}
            return
        d["ball_y"].append(float(ball_xy[1]))
        d["player_path"].append((holder["cx"], holder["cy"]))

    def _close_dribble(self, frame_n, closed):
        d, self._dribble = self._dribble, None
        if d is None or len(d["ball_y"]) < N_DRIBBLE_MIN:
            return
        y = np.array(d["ball_y"])
        dy = np.diff(y)
        sign_flips = int(np.sum(np.abs(np.diff(np.sign(dy[np.abs(dy) > 0.5]))) > 0))
        amplitude = float(y.max() - y.min())
        p0, p1 = np.array(d["player_path"][0]), np.array(d["player_path"][-1])
        moved = float(np.linalg.norm(p1 - p0))
        if sign_flips >= 2 and amplitude >= A_BOUNCE_MIN and moved >= D_MOVE_MIN:
            closed.append(ActionSegment(
                "dribble", d["start_frame"], frame_n, d["tid"], None,
                d["team"], list(d["player_path"]),
                {"bounces": sign_flips // 2, "amplitude_px": amplitude}))

    # ── off-ball movement ────────────────────────────────────────────────────
    def _movement_segments(self, frame_n, window: int = 25) -> list[ActionSegment]:
        out = []
        for tid, hist in self._player_hist.items():
            if tid == self._possessor or len(hist) < 2:
                continue
            recent = [h for h in hist if frame_n - h[0] <= window]
            if len(recent) < 2:
                continue
            p0 = np.array(recent[0][1:])
            p1 = np.array(recent[-1][1:])
            if float(np.linalg.norm(p1 - p0)) >= D_MOVE_MIN:
                out.append(ActionSegment(
                    "move", recent[0][0], frame_n, tid, None, None,
                    [tuple(p0), tuple(p1)]))
        return out
