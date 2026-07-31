"""
Handball Technique Detector — real-time action recognition from tracks.

Detects these handball-specific techniques using pure kinematics
(bounding-box positions + ball trajectory).  No extra model needed.

Techniques detected
───────────────────
  jump_shot       — player airborne while releasing ball toward goal
  standing_shot   — stationary/low shot toward goal
  fast_break      — high-speed run toward opponent goal in open space
  pass            — ball transferred rapidly to another player
  overhead_pass   — ball arcs upward when leaving player
  dribble_run     — player carrying ball over >3 m at medium speed
  gk_save         — goalkeeper intercepts ball heading for goal
  block           — defender deflects incoming ball
  pivot_receive   — player in 6 m zone receives a pass

Each detected event is a TechniqueEvent dataclass and is pushed
to the API server's event ring for AI (Ollama) analysis.

Architecture
────────────
  • Per-track sliding windows (deque) of (cx, cy, frame_n)
  • Ball sliding window for trajectory estimation
  • One TechniqueDetector instance per pipeline run
  • update() called once per processed frame — O(P²) worst-case
    (P = players on court, max 14) — negligible vs YOLO inference

Court coordinate system (metres)
──────────────────────────────────
  (0,0) = left goal center  |  (40,20) = right goal center
  Requires court_det.is_calibrated == True for metre-space checks.
  Falls back to pixel-fraction heuristics when uncalibrated.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── TechniqueEvent ────────────────────────────────────────────────────────────

@dataclass
class TechniqueEvent:
    technique:  str          # see TECHNIQUE_LABELS
    team:       str          # "team_a" | "team_b" | "unknown"
    track_id:   int
    player_id:  str          # "" if not yet labelled
    zone:       str          # court zone string (e.g. "9m_left")
    confidence: float        # 0.0 – 1.0
    frame_n:    int
    ts:         float = field(default_factory=time.perf_counter)
    data:       dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "technique":  self.technique,
            "team":       self.team,
            "track_id":   self.track_id,
            "player_id":  self.player_id,
            "zone":       self.zone,
            "confidence": round(self.confidence, 2),
            "frame_n":    self.frame_n,
            "ts":         round(self.ts, 3),
            **self.data,
        }


TECHNIQUE_LABELS = {
    "jump_shot":      "Jump Shot",
    "standing_shot":  "Standing Shot",
    "fast_break":     "Fast Break Run",
    "pass":           "Pass",
    "overhead_pass":  "Overhead Pass",
    "dribble_run":    "Dribble Run",
    "gk_save":        "Goalkeeper Save",
    "block":          "Block / Intercept",
    "pivot_receive":  "Pivot Receive",
}


# ── Constants ─────────────────────────────────────────────────────────────────

# Pixel-space thresholds (for uncalibrated courts, assuming 1280×720)
_PX_BALL_NEAR   = 80    # px — ball is "near" a player if within this distance
_PX_SHOT_SPEED  = 12    # px/frame — ball moving this fast toward goal = shot
_PX_PASS_SPEED  = 8     # px/frame — ball moving this fast to another player = pass
_PX_SPRINT_SPD  = 9     # px/frame — player sprint threshold

# Metre-space thresholds (used when calibrated)
_M_BALL_NEAR    = 1.8   # m
_M_SHOT_SPEED   = 12.0  # m/s
_M_PASS_SPEED   = 7.0   # m/s
_M_SPRINT_SPD   = 5.5   # m/s
_M_GK_SAVE_DIST = 3.0   # m — GK displacement to count as a save effort
_M_6M_ZONE      = 6.5   # m from goal — pivot receive zone threshold

# Debounce: don't fire same technique for same track within N frames
_DEBOUNCE_FRAMES = {
    "jump_shot":     15,
    "standing_shot": 18,
    "fast_break":    45,
    "pass":          10,
    "overhead_pass": 10,
    "dribble_run":   40,
    "gk_save":       12,
    "block":         12,
    "pivot_receive": 20,
}

_HISTORY_LEN = 40    # frames of position history per track
_BALL_HIS    = 25    # frames of ball history


# ── TechniqueDetector ─────────────────────────────────────────────────────────

class TechniqueDetector:
    """
    Call update() once per frame after tracking.
    Returns a (possibly empty) list of TechniqueEvent objects.
    """

    def __init__(self, fps: float = 25.0):
        self._fps    = max(fps, 1.0)
        # {track_id: deque[(cx_px, cy_px, xm, ym, frame_n)]}
        self._pos:   dict[int, deque] = {}
        # {track_id: last_frame_n fired per technique}
        self._last:  dict[int, dict[str, int]] = {}
        # ball history: deque[(cx_px, cy_px, xm, ym, frame_n)]
        self._ball:  deque = deque(maxlen=_BALL_HIS)
        # track_id of last player who had ball near them
        self._last_possessor: Optional[int] = None
        self._last_possessor_frame: int = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        tracks:     list,
        ball_track,
        court_det,
        frame_n:    int,
        fps:        float,
        pid_map:    dict[int, str],
    ) -> list[TechniqueEvent]:
        """
        Analyse the current frame.  Returns new TechniqueEvent objects only.

        Parameters
        ──────────
        tracks     : list of Track objects (players + ball, from tracker)
        ball_track : Track | None
        court_det  : CourtDetector | None
        frame_n    : current frame counter
        fps        : pipeline FPS (used for speed calculations)
        pid_map    : {track_id: player_id} from TrackBridge
        """
        self._fps = max(fps, 1.0)
        calib = court_det is not None and getattr(court_det, "is_calibrated", False)

        # ── Update position histories ─────────────────────────────────────────
        for t in tracks:
            if t.is_ball:
                continue
            cx = (t.x1 + t.x2) / 2
            cy = (t.y1 + t.y2) / 2
            xm = ym = None
            if calib:
                r = court_det.pixel_to_meter(cx, cy)
                if r:
                    xm, ym = r
            if t.track_id not in self._pos:
                self._pos[t.track_id] = deque(maxlen=_HISTORY_LEN)
                self._last[t.track_id] = {}
            self._pos[t.track_id].append((cx, cy, xm, ym, frame_n))

        # ── Update ball history ───────────────────────────────────────────────
        if ball_track is not None:
            bcx = (ball_track.x1 + ball_track.x2) / 2
            bcy = (ball_track.y1 + ball_track.y2) / 2
            bxm = bym = None
            if calib:
                r = court_det.pixel_to_meter(bcx, bcy)
                if r:
                    bxm, bym = r
            self._ball.append((bcx, bcy, bxm, bym, frame_n))

        events: list[TechniqueEvent] = []
        players = [t for t in tracks if not t.is_ball]

        # ── Detectors ─────────────────────────────────────────────────────────
        events += self._detect_shots(players, ball_track, court_det, calib,
                                     frame_n, pid_map)
        events += self._detect_fast_break(players, court_det, calib,
                                           frame_n, pid_map)
        events += self._detect_pass(players, ball_track, calib,
                                     frame_n, pid_map)
        events += self._detect_dribble(players, ball_track, calib,
                                        frame_n, pid_map)
        events += self._detect_gk_save(players, ball_track, court_det, calib,
                                        frame_n, pid_map)
        events += self._detect_block(players, ball_track, calib,
                                      frame_n, pid_map)
        events += self._detect_pivot_receive(players, ball_track, court_det,
                                              calib, frame_n, pid_map)

        # Purge histories for tracks that have disappeared
        active = {t.track_id for t in players}
        stale  = [tid for tid in self._pos if tid not in active]
        if len(stale) > 30:   # only purge in bulk to avoid per-frame overhead
            for tid in stale:
                self._pos.pop(tid, None)
                self._last.pop(tid, None)

        return events

    # ── Shot detection ────────────────────────────────────────────────────────

    def _detect_shots(self, players, ball_track, court_det, calib,
                      frame_n, pid_map) -> list[TechniqueEvent]:
        """Detects jump shots and standing shots."""
        events = []
        if ball_track is None or len(self._ball) < 5:
            return events

        # Ball velocity over last 5 frames
        b_old = self._ball[-5]
        b_new = self._ball[-1]
        d_frames = max(b_new[4] - b_old[4], 1)

        if calib and b_new[2] is not None and b_old[2] is not None:
            # Metre-space velocity
            vx = (b_new[2] - b_old[2]) / d_frames * self._fps
            vy_m = (b_new[3] - b_old[3]) / d_frames * self._fps
            ball_speed = math.hypot(vx, vy_m)
            threshold  = _M_SHOT_SPEED
            # Is ball heading toward a goal? x<3 (left goal) or x>37 (right goal)
            going_left  = b_new[2] < 5  and vx < -3
            going_right = b_new[2] > 35 and vx >  3
            toward_goal = going_left or going_right
        else:
            # Pixel-space fallback
            vx_px = (b_new[0] - b_old[0]) / d_frames
            vy_px = (b_new[1] - b_old[1]) / d_frames
            ball_speed = math.hypot(vx_px, vy_px)
            threshold  = _PX_SHOT_SPEED
            # Pixel heading: left third or right third of frame
            fw = 1280
            going_left  = b_new[0] < fw * 0.12 and vx_px < -2
            going_right = b_new[0] > fw * 0.88 and vx_px >  2
            toward_goal = going_left or going_right

        if ball_speed < threshold or not toward_goal:
            return events

        # Find nearest player to ball in last 8 frames (the shooter)
        shooter = self._nearest_to_ball_recent(players, frames_back=8,
                                               calib=calib)
        if shooter is None:
            return events

        # Determine if it was a jump shot: shooter's cy decreased then stayed
        is_jump = self._was_jumping(shooter.track_id, frame_n, window=18)

        tech = "jump_shot" if is_jump else "standing_shot"
        conf = min(0.95, ball_speed / (threshold * 2))

        ev = self._make_event(tech, shooter, frame_n, pid_map, calib,
                              court_det, conf,
                              data={"ball_speed": round(ball_speed, 2),
                                    "toward_goal": toward_goal})
        if ev:
            events.append(ev)
        return events

    # ── Fast break detection ──────────────────────────────────────────────────

    def _detect_fast_break(self, players, court_det, calib,
                            frame_n, pid_map) -> list[TechniqueEvent]:
        events = []
        for t in players:
            hist = self._pos.get(t.track_id)
            if not hist or len(hist) < 15:
                continue

            # Speed over last 15 frames
            old = hist[-15]
            new = hist[-1]
            d_f = max(new[4] - old[4], 1)

            if calib and new[2] is not None and old[2] is not None:
                dx = new[2] - old[2]
                dy = new[3] - old[3]
                speed = math.hypot(dx, dy) / d_f * self._fps
                threshold = _M_SPRINT_SPD
                # Is player moving toward their own attack half?
                in_open = self._is_in_open_space(t, players, calib, thresh_m=3.0)
            else:
                dx = new[0] - old[0]
                dy = new[1] - old[1]
                speed = math.hypot(dx, dy) / d_f
                threshold = _PX_SPRINT_SPD
                in_open = self._is_in_open_space(t, players, calib=False,
                                                  thresh_px=120)

            if speed < threshold or not in_open:
                continue

            conf = min(0.92, speed / (threshold * 1.6))
            ev = self._make_event("fast_break", t, frame_n, pid_map, calib,
                                  court_det, conf,
                                  data={"speed": round(speed, 2)})
            if ev:
                events.append(ev)
        return events

    # ── Pass detection ────────────────────────────────────────────────────────

    def _detect_pass(self, players, ball_track, calib,
                     frame_n, pid_map) -> list[TechniqueEvent]:
        events = []
        if ball_track is None or len(self._ball) < 8:
            return events

        b_old = self._ball[-8]
        b_new = self._ball[-1]
        d_f   = max(b_new[4] - b_old[4], 1)

        if calib and b_new[2] is not None and b_old[2] is not None:
            speed = math.hypot(b_new[2]-b_old[2], b_new[3]-b_old[3]) / d_f * self._fps
            threshold = _M_PASS_SPEED
        else:
            speed = math.hypot(b_new[0]-b_old[0], b_new[1]-b_old[1]) / d_f
            threshold = _PX_PASS_SPEED

        if speed < threshold:
            return events

        # Ball must be heading toward a player (not toward goal)
        receiver = self._nearest_to_ball_current(players, ball_track, calib)
        sender   = self._last_possessor
        if receiver is None or (sender is not None and receiver.track_id == sender):
            return events

        # Sender must be different from receiver and had ball recently
        sender_track = next((t for t in players
                             if t.track_id == sender), None) if sender else None
        if sender_track is None:
            return events

        # Ball must travel BETWEEN two players (not out to goal area)
        if not self._ball_between_players(b_old, b_new, sender_track, receiver):
            return events

        is_overhead = self._is_overhead_arc()
        tech = "overhead_pass" if is_overhead else "pass"
        conf = min(0.90, speed / (threshold * 1.8))

        ev = self._make_event(tech, sender_track, frame_n, pid_map, calib,
                              None, conf,
                              data={"receiver_id": receiver.track_id,
                                    "ball_speed": round(speed, 2)})
        if ev:
            events.append(ev)

        # Update possessor
        self._last_possessor       = receiver.track_id
        self._last_possessor_frame = frame_n
        return events

    # ── Dribble run detection ─────────────────────────────────────────────────

    def _detect_dribble(self, players, ball_track, calib,
                         frame_n, pid_map) -> list[TechniqueEvent]:
        events = []
        if ball_track is None:
            return events

        # Player and ball must stay close over an extended run (25 frames)
        for t in players:
            hist = self._pos.get(t.track_id)
            if not hist or len(hist) < 25:
                continue
            if not self._ball_stayed_near_player(t.track_id, calib, frames=25):
                continue
            # Distance covered
            old = hist[-25]; new = hist[-1]
            d_f = max(new[4] - old[4], 1)
            if calib and new[2] is not None and old[2] is not None:
                dist  = math.hypot(new[2]-old[2], new[3]-old[3])
                moved = dist > 3.0
                conf  = min(0.85, dist / 6.0)
            else:
                dist  = math.hypot(new[0]-old[0], new[1]-old[1])
                moved = dist > 80
                conf  = min(0.85, dist / 180)

            if not moved:
                continue
            ev = self._make_event("dribble_run", t, frame_n, pid_map, calib,
                                  None, conf,
                                  data={"dist": round(dist, 2)})
            if ev:
                events.append(ev)
        return events

    # ── GK save detection ─────────────────────────────────────────────────────

    def _detect_gk_save(self, players, ball_track, court_det, calib,
                         frame_n, pid_map) -> list[TechniqueEvent]:
        events = []
        if ball_track is None or len(self._ball) < 6:
            return events

        goalkeepers = [t for t in players
                       if t.team in ("goalkeeper", "team_a", "team_b")
                       and self._is_near_goal(t, calib, court_det)]
        if not goalkeepers:
            return events

        for gk in goalkeepers:
            hist = self._pos.get(gk.track_id)
            if not hist or len(hist) < 6:
                continue
            # GK rapid lateral/vertical displacement
            old = hist[-6]; new = hist[-1]
            if calib and new[2] is not None and old[2] is not None:
                disp = math.hypot(new[2]-old[2], new[3]-old[3])
                moved = disp > 1.2
            else:
                disp = math.hypot(new[0]-old[0], new[1]-old[1])
                moved = disp > 40

            if not moved:
                continue

            # Ball must have been heading toward goal and is now deflected
            b_changed = self._ball_direction_changed(frames_back=6)
            if not b_changed:
                continue

            conf = min(0.88, disp / (_M_GK_SAVE_DIST if calib else 120))
            ev = self._make_event("gk_save", gk, frame_n, pid_map, calib,
                                  court_det, conf,
                                  data={"gk_disp": round(disp, 2)})
            if ev:
                events.append(ev)
        return events

    # ── Block detection ───────────────────────────────────────────────────────

    def _detect_block(self, players, ball_track, calib,
                       frame_n, pid_map) -> list[TechniqueEvent]:
        events = []
        if ball_track is None or len(self._ball) < 5:
            return events

        if not self._ball_direction_changed(frames_back=5):
            return events

        # Who caused the deflection? Nearest player to the ball at the change point
        deflector = self._nearest_to_ball_recent(players, frames_back=5,
                                                  calib=calib)
        if deflector is None:
            return events

        # Must be a defender (opponent of the shooter/possessor)
        if (self._last_possessor is not None
                and deflector.track_id == self._last_possessor):
            return events

        ev = self._make_event("block", deflector, frame_n, pid_map, calib,
                              None, 0.78)
        if ev:
            events.append(ev)
        return events

    # ── Pivot receive detection ───────────────────────────────────────────────

    def _detect_pivot_receive(self, players, ball_track, court_det, calib,
                               frame_n, pid_map) -> list[TechniqueEvent]:
        events = []
        if ball_track is None:
            return events

        for t in players:
            if not self._is_in_6m_zone(t, calib, court_det):
                continue
            # Ball must be close to this player now and wasn't 8 frames ago
            near_now = self._ball_near_player(t, calib, ball_track)
            if not near_now:
                continue
            hist = self._pos.get(t.track_id)
            if hist and len(hist) >= 8:
                # Was the ball NOT near 8 frames ago? (it was received, not carried)
                far_before = not self._was_ball_near_player_at(t.track_id, calib,
                                                                 frame_n - 8)
                if not far_before:
                    continue
            ev = self._make_event("pivot_receive", t, frame_n, pid_map, calib,
                                  court_det, 0.82)
            if ev:
                events.append(ev)
        return events

    # ── Helper methods ────────────────────────────────────────────────────────

    def _make_event(self, tech: str, track, frame_n: int,
                    pid_map: dict, calib: bool, court_det,
                    conf: float, data: dict = None) -> Optional[TechniqueEvent]:
        """Apply debounce and build TechniqueEvent."""
        tid  = track.track_id
        last = self._last.setdefault(tid, {})
        debounce = _DEBOUNCE_FRAMES.get(tech, 20)
        if frame_n - last.get(tech, -9999) < debounce:
            return None
        last[tech] = frame_n

        zone = self._zone(track, calib, court_det)
        pid  = pid_map.get(tid, "")
        return TechniqueEvent(
            technique=tech,
            team=track.team,
            track_id=tid,
            player_id=pid,
            zone=zone,
            confidence=round(conf, 2),
            frame_n=frame_n,
            data=data or {},
        )

    def _nearest_to_ball_recent(self, players, frames_back: int,
                                 calib: bool) -> Optional[object]:
        """Find the player closest to the ball over the last N frames."""
        if not self._ball:
            return None
        # Use ball positions from frames_back frames ago
        b = self._ball[max(0, len(self._ball) - frames_back)]
        return self._nearest_player_to_point(players, b[0], b[1],
                                              b[2], b[3], calib)

    def _nearest_to_ball_current(self, players, ball_track,
                                  calib: bool) -> Optional[object]:
        if not self._ball:
            return None
        b = self._ball[-1]
        return self._nearest_player_to_point(players, b[0], b[1],
                                              b[2], b[3], calib)

    def _nearest_player_to_point(self, players, px, py, mx, my,
                                  calib: bool) -> Optional[object]:
        best = None
        best_d = float("inf")
        for t in players:
            hist = self._pos.get(t.track_id)
            if not hist:
                continue
            p = hist[-1]
            if calib and mx is not None and p[2] is not None:
                d = math.hypot(p[2]-mx, p[3]-my)
                thresh = _M_BALL_NEAR
            else:
                d = math.hypot(p[0]-px, p[1]-py)
                thresh = _PX_BALL_NEAR
            if d < best_d and d < thresh * 2:
                best_d = d
                best   = t
        return best

    def _was_jumping(self, tid: int, frame_n: int, window: int) -> bool:
        """Return True if track_id's cy decreased then stabilised within window."""
        hist = self._pos.get(tid)
        if not hist or len(hist) < window:
            return False
        cys = [h[1] for h in list(hist)[-window:]]
        # Find minimum (highest point in frame = lowest cy)
        min_i = cys.index(min(cys))
        # Must have been rising (cy decreasing) before midpoint
        if min_i < 3 or min_i > len(cys) - 3:
            return False
        rise = cys[0] - cys[min_i]
        return rise > 8   # at least 8 pixels upward arc

    def _ball_near_player(self, track, calib: bool, ball_track) -> bool:
        hist = self._pos.get(track.track_id)
        if not hist or not self._ball:
            return False
        p = hist[-1]; b = self._ball[-1]
        if calib and p[2] is not None and b[2] is not None:
            return math.hypot(p[2]-b[2], p[3]-b[3]) < _M_BALL_NEAR
        return math.hypot(p[0]-b[0], p[1]-b[1]) < _PX_BALL_NEAR

    def _was_ball_near_player_at(self, tid: int, calib: bool,
                                   target_frame: int) -> bool:
        hist = self._pos.get(tid)
        if not hist or not self._ball:
            return False
        # Find position at target frame
        p = None
        for h in hist:
            if h[4] == target_frame:
                p = h; break
        b = None
        for bh in self._ball:
            if bh[4] == target_frame:
                b = bh; break
        if p is None or b is None:
            return False
        if calib and p[2] is not None and b[2] is not None:
            return math.hypot(p[2]-b[2], p[3]-b[3]) < _M_BALL_NEAR
        return math.hypot(p[0]-b[0], p[1]-b[1]) < _PX_BALL_NEAR

    def _ball_stayed_near_player(self, tid: int, calib: bool,
                                   frames: int) -> bool:
        hist = self._pos.get(tid)
        if not hist or len(hist) < frames or len(self._ball) < frames:
            return False
        ph = list(hist)[-frames:]
        bh = list(self._ball)[-frames:]
        near_count = 0
        for p, b in zip(ph, bh):
            if calib and p[2] is not None and b[2] is not None:
                d = math.hypot(p[2]-b[2], p[3]-b[3])
                near_count += d < _M_BALL_NEAR
            else:
                d = math.hypot(p[0]-b[0], p[1]-b[1])
                near_count += d < _PX_BALL_NEAR
        return near_count > frames * 0.65

    def _ball_direction_changed(self, frames_back: int) -> bool:
        """Detect if the ball's velocity direction has changed sharply."""
        if len(self._ball) < frames_back + 2:
            return False
        def vel(a, b):
            df = max(b[4]-a[4], 1)
            return (b[0]-a[0])/df, (b[1]-a[1])/df
        mid  = max(0, len(self._ball) - frames_back)
        vx1, vy1 = vel(self._ball[mid-1] if mid>0 else self._ball[0],
                       self._ball[mid])
        vx2, vy2 = vel(self._ball[-2], self._ball[-1])
        dot = vx1*vx2 + vy1*vy2
        m1  = math.hypot(vx1, vy1)
        m2  = math.hypot(vx2, vy2)
        if m1 < 0.5 or m2 < 0.5:
            return False
        cosine = dot / (m1 * m2)
        return cosine < 0.3   # angle > ~72°

    def _is_in_open_space(self, track, players, calib: bool,
                           thresh_m: float = 3.0,
                           thresh_px: float = 120) -> bool:
        """True if no opponent within threshold distance."""
        hist = self._pos.get(track.track_id)
        if not hist:
            return False
        p = hist[-1]
        for t in players:
            if t.track_id == track.track_id or t.team == track.team:
                continue
            oh = self._pos.get(t.track_id)
            if not oh:
                continue
            o = oh[-1]
            if calib and p[2] is not None and o[2] is not None:
                if math.hypot(p[2]-o[2], p[3]-o[3]) < thresh_m:
                    return False
            else:
                if math.hypot(p[0]-o[0], p[1]-o[1]) < thresh_px:
                    return False
        return True

    def _ball_between_players(self, b_old, b_new, sender, receiver) -> bool:
        """True if the ball trajectory goes from sender toward receiver."""
        sh = self._pos.get(sender.track_id)
        rh = self._pos.get(receiver.track_id)
        if not sh or not rh:
            return False
        s = sh[-1]; r = rh[-1]
        # Vector from sender to receiver
        sr_x = r[0] - s[0]; sr_y = r[1] - s[1]
        # Vector of ball movement
        bv_x = b_new[0] - b_old[0]; bv_y = b_new[1] - b_old[1]
        dot = sr_x*bv_x + sr_y*bv_y
        return dot > 0   # same general direction

    def _is_overhead_arc(self) -> bool:
        """Ball went upward (y decreased) then forward — overhead pass arc."""
        if len(self._ball) < 10:
            return False
        mid = len(self._ball) // 2
        cys = [b[1] for b in self._ball]
        return cys[mid] < cys[0] - 15 and cys[mid] < cys[-1] - 15

    def _is_near_goal(self, track, calib: bool, court_det) -> bool:
        hist = self._pos.get(track.track_id)
        if not hist:
            return False
        p = hist[-1]
        if calib and p[2] is not None:
            return p[2] < 5 or p[2] > 35
        # Pixel fallback: near left or right edge
        return p[0] < 100 or p[0] > 1180

    def _is_in_6m_zone(self, track, calib: bool, court_det) -> bool:
        hist = self._pos.get(track.track_id)
        if not hist:
            return False
        p = hist[-1]
        if calib and p[2] is not None:
            # Near either goal within 6m zone
            dist_left  = math.hypot(p[2], p[3] - 10.0)
            dist_right = math.hypot(p[2] - 40.0, p[3] - 10.0)
            return dist_left < _M_6M_ZONE or dist_right < _M_6M_ZONE
        return p[0] < 160 or p[0] > 1120

    def _zone(self, track, calib: bool, court_det) -> str:
        hist = self._pos.get(track.track_id)
        if not hist:
            return "unknown"
        p = hist[-1]
        if calib and p[2] is not None:
            xm, ym = p[2], p[3]
            if   xm < 6:   return "6m_left"
            elif xm < 9:   return "9m_left"
            elif xm < 18:  return "back_left"
            elif xm < 22:  return "centre"
            elif xm < 31:  return "back_right"
            elif xm < 34:  return "9m_right"
            else:           return "6m_right"
        # Pixel fallback (1280-wide frame)
        x = p[0]
        if   x < 150:  return "6m_left"
        elif x < 280:  return "9m_left"
        elif x < 500:  return "back_left"
        elif x < 780:  return "centre"
        elif x < 1000: return "back_right"
        elif x < 1130: return "9m_right"
        else:           return "6m_right"
