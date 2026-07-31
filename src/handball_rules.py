"""
Handball Rules Engine — Professional Edition
Supports: Passes, Shots, Goals, Saves, Missed, Fouls, Fast Breaks.
Physics-based ball filtering and interpolation.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Constants
PERIOD_DURATION_S      = 30 * 60
PASSIVE_WARNING_S      = 25.0
PASSIVE_VIOLATION_S    = 35.0
BALL_HISTORY_LEN       = 20

POSSESSION_RADIUS_PX   = 150      # how close the ball must be to a player to be "possessed"
PASS_SPEED_MIN_PX      = 8        # pixels/frame – lower to detect slow passes
SHOT_SPEED_MIN_PX      = 18       # pixels/frame – lower to catch weak shots
SHOT_DEBOUNCE_S        = 1.2
GOAL_COOLDOWN_FRAMES   = 75

BRIDGE_RADIUS_M        = 1.5
BRIDGE_PROXY_PER_H     = 1.5
BRIDGE_WINDOW_FRAMES   = 12

class EventType(str, Enum):
    POSSESSION_CHANGE = "possession_change"
    PASS              = "pass"
    SHOT              = "shot"
    GOAL              = "goal"
    SAVE              = "save"
    MISSED_OUT        = "missed_out"
    FAST_BREAK        = "fast_break"
    FOUL              = "foul"
    PASSIVE_WARNING   = "passive_warning"
    PASSIVE_VIOLATION = "passive_violation"
    PERIOD_END        = "period_end"

@dataclass
class MatchEvent:
    event_type: EventType
    team: str
    timestamp: float
    frame_n: int
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event_type.value,
            "team": self.team,
            "ts": round(self.timestamp, 3),
            "frame": self.frame_n,
            **self.data,
        }

class HandballEngine:
    def __init__(self, frame_w: int, frame_h: int, fps: float = 25.0, team_a_attacks_right: bool = True):
        self.fw = frame_w
        self.fh = frame_h
        self.fps = fps
        self._a_attacks_right = team_a_attacks_right

        # Dynamic goal box (set per-frame from goal_post_detector). When None,
        # we fall back to the static pixel rectangle in _make_goal_regions().
        self._dynamic_goal_box = None  # tuple (x_left, y_top, x_right, y_bottom, side)

        # State
        self.score = {"team_a": 0, "team_b": 0}
        self.shots = {"team_a": 0, "team_b": 0}
        self.saves = {"team_a": 0, "team_b": 0}
        self.passes = {"team_a": 0, "team_b": 0}
        self.fouls = {"team_a": 0, "team_b": 0}
        self.fast_breaks = {"team_a": 0, "team_b": 0}

        self.possession: Optional[str] = None
        self.possession_start: Optional[float] = None
        self.possession_duration: float = 0.0
        self._current_player_id: Optional[int] = None   # who holds the ball
        # Play understanding: the chain of players who touched the ball in the
        # current possession  →  P1 → P2 → shooter.  The shooter's predecessor
        # is the ASSIST. Resets when the ball turns over to the other team.
        self._possession_chain: list[int] = []
        self._chain_team: Optional[str] = None
        self._last_passer_id: Optional[int] = None

        # Ball tracking
        self._ball_history: list = []   # (x, y, timestamp, frame)
        self._ball_hist: deque = deque(maxlen=BALL_HISTORY_LEN)
        self._last_valid_ball_pos: Optional[tuple] = None
        self._static_ball_frames = 0

        self._homography: Optional[np.ndarray] = None
        self._period_start_ts: Optional[float] = None
        self._passive_warned = False
        self.period = 1
        self.events: list[MatchEvent] = []

        # For shot resolution
        self._active_shot: Optional[dict] = None   # {"team", "shooter_id", "start_frame", "start_time"}
        self._last_goal_frame = -9999

        # Goal zones (fallback pixel regions)
        self._left_goal, self._right_goal = self._make_goal_regions()

    # -----------------------------------------------------------------
    #  Core public methods
    # -----------------------------------------------------------------
    def set_homography(self, H: Optional[np.ndarray]) -> None:
        self._homography = None if H is None else np.asarray(H, dtype=np.float64)

    def update(self, tracks: list, raw_ball_track, frame_n: int,
               ball_hallucinated: bool = False) -> list[MatchEvent]:
        """Main update call – called every frame from pipeline."""
        now = time.perf_counter()
        if self._period_start_ts is None:
            self._period_start_ts = now

        # Track consecutive phantom frames so we can suppress noisy events.
        # Once the ball has been phantom for >2 consecutive frames we no
        # longer trust its trajectory enough to fire SHOT/PASS — the
        # rule engine still tracks possession, but doesn't generate
        # statistical events that would inflate the team totals.
        if ball_hallucinated:
            self._phantom_streak = getattr(self, "_phantom_streak", 0) + 1
        else:
            self._phantom_streak = 0
        trust_ball = self._phantom_streak <= 2

        # Validate and filter ball
        ball_track = self._validate_ball(raw_ball_track, frame_n)

        # Store ball trajectory (or predict if lost)
        if ball_track:
            self._last_valid_ball_pos = (ball_track.cx, ball_track.cy)
            self._ball_history.append((ball_track.cx, ball_track.cy, now, frame_n))
            self._ball_hist.append((ball_track.cx, ball_track.cy, frame_n, now))
            if len(self._ball_history) > BALL_HISTORY_LEN:
                self._ball_history.pop(0)
        else:
            self._interpolate_ball(frame_n, now)

        new_events: list[MatchEvent] = []
        # Process possession, passes, shots, goals, fast breaks.
        # When the ball trust is gone (long phantom streak), we still update
        # internal state but skip the event-emitting passes that would
        # otherwise corrupt the match stats.
        if trust_ball:
            new_events += self._process_possession_and_passes(tracks, ball_track, frame_n, now)
            new_events += self._process_shots_and_goals(tracks, ball_track, frame_n, now)
            new_events += self._check_fast_break(tracks, frame_n, now)
        else:
            # Cancel any pending shot — we shouldn't resolve it on phantom data
            self._active_shot = None

        # ── Direct goal detection ──
        # Even if no SHOT event was registered (ball lost during the release),
        # if the ball is observed INSIDE the dynamic goal box on REAL detections
        # for N consecutive frames, count it as a goal. This catches the
        # "ball clearly went in but no shot was tracked" case the user reported.
        if not ball_hallucinated:
            new_events += self._check_direct_goal(ball_track, frame_n, now)
            # PERMISSIVE PATH (ported from old stable pipeline):
            # Check the static goal regions (12% of frame width on each side).
            # These are MUCH wider than the dynamic_goal_box and catch goals
            # the post-detector misses.
            new_events += self._check_goal_smart(ball_track, tracks, frame_n, now)

        # Update possession duration for HUD
        if self.possession and self.possession_start:
            self.possession_duration = now - self.possession_start
        else:
            self.possession_duration = 0.0

        self.events.extend(new_events)
        return new_events

    def flip_sides(self) -> None:
        self._a_attacks_right = not self._a_attacks_right
        self.possession = None
        self.possession_start = None
        self._current_player_id = None

    def manual_goal(self, team: str) -> None:
        self.score[team] += 1
        ev = MatchEvent(EventType.GOAL, team, time.perf_counter(), -1, {"manual": True})
        self.events.append(ev)

    @property
    def shot_accuracy(self) -> dict[str, float]:
        """Required by dashboard API."""
        return {team: round(self.score[team] / s * 100, 1) if s > 0 else 0.0
                for team, s in self.shots.items()}

    def hud_data(self) -> dict:
        period_elapsed = time.perf_counter() - self._period_start_ts if self._period_start_ts else 0.0
        return {
            "score": self.score.copy(),
            "period": self.period,
            "period_elapsed_s": round(period_elapsed, 0),
            "possession": self.possession,
            "possession_duration_s": round(self.possession_duration, 1),
            "possession_chain": list(self._possession_chain),   # P1→P2→… buildup
            "shots": self.shots.copy(),
            "saves": self.saves.copy(),
            "fast_breaks": self.fast_breaks.copy(),
            "passes": self.passes.copy(),
            "shot_accuracy": self.shot_accuracy,
            "ball_speed_px": 0.0,    # can be computed from _ball_hist if needed
        }

    def recent_events_text(self, n: int = 6) -> list[str]:
        return [f"[{ev.team.replace('team_', 'T').upper()}] {ev.event_type.value.replace('_',' ').upper()}"
                for ev in reversed(self.events[-n:])]

    # -----------------------------------------------------------------
    #  Private helpers
    # -----------------------------------------------------------------
    def _validate_ball(self, ball_track, frame_n: int):
        """Reject false positives: wrong size, wrong aspect ratio, teleporting, static markers."""
        if ball_track is None:
            return None

        # 1. Size filter
        try:
            w = ball_track.x2 - ball_track.x1
            h = ball_track.y2 - ball_track.y1
            if w > 150 or h > 150:
                return None
            if w > 0 and h > 0:
                ratio = w / h
                if ratio > 2.5 or ratio < 0.4:
                    return None
        except AttributeError:
            pass

        # 2. Teleportation (more than 300 px per frame – impossible)
        if self._last_valid_ball_pos is not None:
            dist = math.hypot(ball_track.cx - self._last_valid_ball_pos[0],
                              ball_track.cy - self._last_valid_ball_pos[1])
            if dist > 300:
                # Accept but clear history to avoid wrong shot detection
                self._ball_history.clear()
                self._ball_hist.clear()

        # 3. Static marker (e.g., penalty spot, logo)
        if self._last_valid_ball_pos is not None:
            dist = math.hypot(ball_track.cx - self._last_valid_ball_pos[0],
                              ball_track.cy - self._last_valid_ball_pos[1])
            if dist < 2.0:
                self._static_ball_frames += 1
            else:
                self._static_ball_frames = 0
            if self._static_ball_frames > 50:      # ~2 seconds stationary
                return None

        return ball_track

    def _interpolate_ball(self, frame_n: int, now: float):
        """If ball disappeared, continue its trajectory with friction."""
        if len(self._ball_history) < 2:
            return
        last_x, last_y, last_t, last_f = self._ball_history[-1]
        prev_x, prev_y, prev_t, prev_f = self._ball_history[-2]
        frames = max(last_f - prev_f, 1)
        vx = (last_x - prev_x) / frames
        vy = (last_y - prev_y) / frames
        # friction (air resistance)
        vx *= 0.85
        vy *= 0.85
        if abs(vx) < 0.5 and abs(vy) < 0.5:
            return
        new_x = last_x + vx
        new_y = last_y + vy
        self._ball_history.append((new_x, new_y, now, frame_n))
        self._ball_hist.append((new_x, new_y, frame_n, now))
        if len(self._ball_history) > BALL_HISTORY_LEN:
            self._ball_history.pop(0)
        # create synthetic ball object (duck typing)
        class SynthBall:
            pass
        synth = SynthBall()
        synth.cx, synth.cy = new_x, new_y
        synth.x1, synth.y1 = new_x-5, new_y-5
        synth.x2, synth.y2 = new_x+5, new_y+5
        synth.class_id = 32
        return synth

    # Possession is sticky: a candidate must be the closest player for
    # POSSESSION_DEBOUNCE_FRAMES consecutive frames before we accept the flip.
    # Without this, ball jitter between phantom positions makes possession
    # change on almost every frame (the user reported "team_a→b→a→b→a in 1s").
    POSSESSION_DEBOUNCE_FRAMES = 6

    def _push_chain(self, pid, team):
        """Build the possession chain  P1 → P2 → … (the play's buildup)."""
        if team != self._chain_team:                 # turnover → new possession
            self._possession_chain = []
            self._chain_team = team
        if not self._possession_chain or self._possession_chain[-1] != pid:
            self._possession_chain.append(pid)
            if len(self._possession_chain) > 12:
                self._possession_chain = self._possession_chain[-12:]

    def assist_for(self, shooter_pid):
        """The player who passed to the shooter = the predecessor in the chain."""
        c = self._possession_chain
        if shooter_pid in c:
            i = c.index(shooter_pid)
            if i > 0:
                return c[i - 1]
        elif len(c) >= 2:
            return c[-2]                              # shooter not yet appended
        return None

    def _process_possession_and_passes(self, tracks, ball_track, frame_n, now):
        events = []
        if not ball_track:
            return events

        # Find closest player to ball
        closest = None
        min_dist = POSSESSION_RADIUS_PX
        for t in tracks:
            if getattr(t, 'class_id', 0) == 32:   # skip ball itself
                continue
            d = math.hypot(t.cx - ball_track.cx, t.cy - ball_track.cy)
            if d < min_dist:
                min_dist = d
                closest = t

        if closest is None:
            return events

        cand_team = getattr(closest, 'team', None)
        if cand_team not in ("team_a", "team_b"):
            cand_team = self.possession if self.possession else "team_a"

        cand_pid = closest.track_id

        # ── Debounce: only flip possession after sustained closeness ──
        # Track the candidate via a small streak counter. If the same team
        # remains closest for POSSESSION_DEBOUNCE_FRAMES frames, commit the flip.
        pending = getattr(self, "_pending_possession", None)
        if pending is None or pending.get("team") != cand_team:
            self._pending_possession = {
                "team":     cand_team,
                "pid":      cand_pid,
                "streak":   1,
                "since_fn": frame_n,
            }
            # Don't act on possession change yet — return without firing event
            # (still update _current_player_id so passes within same team work)
            if self.possession == cand_team:
                # Same team but new player → potential pass
                if self._current_player_id is not None and self._current_player_id != cand_pid:
                    if len(self._ball_hist) >= 2:
                        bx1, by1, _, _ = self._ball_hist[-2]
                        bx2, by2, _, _ = self._ball_hist[-1]
                        speed = math.hypot(bx2 - bx1, by2 - by1)
                        if speed > PASS_SPEED_MIN_PX:
                            self.passes[cand_team] += 1
                            events.append(MatchEvent(EventType.PASS, cand_team, now, frame_n, {
                                "from_id": self._current_player_id,
                                "to_id":   cand_pid,
                                "speed_px": round(speed, 1),
                            }))
                self._current_player_id = cand_pid
                self._push_chain(cand_pid, cand_team)
            return events

        # Same candidate team as last frame → grow the streak
        pending["streak"] += 1
        pending["pid"] = cand_pid

        # Hasn't reached the debounce threshold yet — keep waiting
        if pending["streak"] < self.POSSESSION_DEBOUNCE_FRAMES:
            return events

        # Confirmed flip
        if self.possession != cand_team:
            old_team = self.possession or "none"
            self.possession = cand_team
            self.possession_start = now
            events.append(MatchEvent(EventType.POSSESSION_CHANGE, cand_team, now, frame_n, {
                "player_id":   cand_pid,
                "to_player_id": cand_pid,
                "from_team":   old_team,
            }))

        # In-team player change → pass detection
        if self._current_player_id is not None and self._current_player_id != cand_pid:
            if len(self._ball_hist) >= 2:
                bx1, by1, _, _ = self._ball_hist[-2]
                bx2, by2, _, _ = self._ball_hist[-1]
                speed = math.hypot(bx2 - bx1, by2 - by1)
                if speed > PASS_SPEED_MIN_PX:
                    self.passes[cand_team] += 1
                    events.append(MatchEvent(EventType.PASS, cand_team, now, frame_n, {
                        "from_id":  self._current_player_id,
                        "to_id":    cand_pid,
                        "speed_px": round(speed, 1),
                    }))

        self._current_player_id = cand_pid
        self._push_chain(cand_pid, cand_team)
        return events

    # Hard physical caps: anything above is a phantom-jump artefact.
    SHOT_SPEED_MAX_PX        = 90.0    # >90 px/frame is teleport, not a shot
    SHOT_GLOBAL_COOLDOWN_S   = 1.5     # min time between any two SHOT events
    PASS_SPEED_MAX_PX        = 80.0    # >80 px/frame between two ball points = phantom

    def _process_shots_and_goals(self, tracks, ball_track, frame_n, now):
        events = []
        if not ball_track or len(self._ball_hist) < 3:
            # Resolve any pending shot (e.g., after ball lost)
            if self._active_shot:
                self._resolve_shot(tracks, None, now, frame_n, events)
            return events

        # Calculate ball speed (px/frame) between current and 3 frames ago
        _, _, _, bf = self._ball_hist[-1]
        px, py, _, pf = self._ball_hist[-3]
        frames = max(bf - pf, 1)
        speed = math.hypot(ball_track.cx - px, ball_track.cy - py) / frames

        # Sanity cap: speeds above SHOT_SPEED_MAX_PX are physically impossible
        # for a ball at any frame rate — they're phantom jumps. Discard.
        if speed > self.SHOT_SPEED_MAX_PX:
            return events

        # Global cooldown so one attack doesn't generate 5+ "SHOT" events
        last_shot_ts = getattr(self, "_last_shot_ts", -1e9)
        if now - last_shot_ts < self.SHOT_GLOBAL_COOLDOWN_S:
            # Still resolve any pending shot if it timed out
            if self._active_shot:
                self._resolve_shot(tracks, ball_track, now, frame_n, events)
            return events

        # Check if this is a shot (high speed and moving towards opponent goal)
        if speed > SHOT_SPEED_MIN_PX and not self._active_shot and self.possession:
            # Determine attacking direction
            attacking_right = (self.possession == "team_a" and self._a_attacks_right) or \
                              (self.possession == "team_b" and not self._a_attacks_right)
            # ball moving towards goal?
            goal_x = self.fw if attacking_right else 0
            moving_to_goal = (attacking_right and ball_track.cx > px) or \
                             (not attacking_right and ball_track.cx < px)
            if moving_to_goal:
                self.shots[self.possession] += 1
                self._active_shot = {
                    "team": self.possession,
                    "shooter_id": self._current_player_id,
                    "start_time": now,
                    "start_frame": frame_n,
                }
                self._last_shot_ts = now
                events.append(MatchEvent(EventType.SHOT, self.possession, now, frame_n, {
                    "shooter_id": self._current_player_id,
                    "speed_px": round(speed, 1),
                    "x": ball_track.cx,
                    "y": ball_track.cy
                }))

        # Resolve active shot (goal, save, missed)
        if self._active_shot:
            self._resolve_shot(tracks, ball_track, now, frame_n, events)

        return events

    def _resolve_shot(self, tracks, ball_track, now, frame_n, events):
        if not self._active_shot:
            return
        shot = self._active_shot
        team = shot["team"]
        opp_team = "team_b" if team == "team_a" else "team_a"

        # Timeout: if more than 3 seconds have passed, consider missed
        if now - shot["start_time"] > 3.0:
            events.append(MatchEvent(EventType.MISSED_OUT, team, now, frame_n,
                                     {"shooter_id": shot["shooter_id"]}))
            self._active_shot = None
            return

        # If ball is held by opponent near their goal → Save
        if ball_track is None:
            # No ball, cannot decide yet
            return

        # Find closest player to ball
        closest = None
        min_dist = POSSESSION_RADIUS_PX
        for t in tracks:
            if getattr(t, 'class_id', 0) == 32:
                continue
            d = math.hypot(t.cx - ball_track.cx, t.cy - ball_track.cy)
            if d < min_dist:
                min_dist = d
                closest = t

        if closest and getattr(closest, 'team', '') == opp_team:
            # Opponent caught the ball → Save (goalkeeper or field player)
            # Check if near goal (heuristic)
            goal_x = self._get_attacking_goal_x(team)
            dist_to_goal = abs(closest.cx - goal_x)
            if dist_to_goal < self.fw * 0.25:   # within quarter of field width
                self.saves[opp_team] += 1
                events.append(MatchEvent(EventType.SAVE, opp_team, now, frame_n, {
                    "gk_id": closest.track_id,
                    "shooter_id": shot["shooter_id"]
                }))
                self._active_shot = None
                return

        # Check if ball crossed goal line
        bx = ball_track.cx
        by = ball_track.cy
        goal_x = self._get_attacking_goal_x(team)

        # ── Dynamic goal box (preferred) — adapts to camera angle ──
        # Debounce: a goal fires only after the ball has been inside the goal
        # frame for ≥ MIN_IN_GOAL_FRAMES consecutive checks. This prevents
        # phantom-ball flickers + single-frame jitter from inflating the score.
        MIN_IN_GOAL_FRAMES = 2
        gb = self._dynamic_goal_box
        if gb is not None:
            try:
                pad = 12.0
                inside_x = (gb.x_left - pad) <= bx <= (gb.x_right + pad)
                inside_y = (gb.y_top  - pad) <= by <= (gb.y_bottom + pad)
                crossed  = inside_x and inside_y
            except AttributeError:
                crossed = (goal_x > self.fw/2 and bx >= self.fw - 30) or \
                          (goal_x < self.fw/2 and bx <= 30)
        else:
            crossed = (goal_x > self.fw/2 and bx >= self.fw - 30) or \
                      (goal_x < self.fw/2 and bx <= 30)

        # Debounce counter
        if crossed:
            self._in_goal_streak = getattr(self, "_in_goal_streak", 0) + 1
        else:
            self._in_goal_streak = 0

        if not crossed:
            is_goal = False
        elif self._in_goal_streak < MIN_IN_GOAL_FRAMES:
            # Inside the goal but not yet confirmed — wait one more frame
            return
        elif gb is not None:
            # Confirmed: ball stayed inside the detected goal box for ≥ N frames
            is_goal = True
        else:
            # Static fallback — also gated by debounce
            is_goal = (self.fh * 0.35) <= by <= (self.fh * 0.65)

        if crossed:
            if is_goal:
                self.score[team] += 1
                events.append(MatchEvent(EventType.GOAL, team, now, frame_n, {
                    "shooter_id": shot["shooter_id"],
                    "side": "right" if goal_x > self.fw/2 else "left"
                }))
            else:
                events.append(MatchEvent(EventType.MISSED_OUT, team, now, frame_n,
                                         {"shooter_id": shot["shooter_id"]}))
            self._active_shot = None

    def _get_attacking_goal_x(self, team: str) -> float:
        attacking_right = (team == "team_a" and self._a_attacks_right) or \
                          (team == "team_b" and not self._a_attacks_right)
        return self.fw if attacking_right else 0.0

    # -----------------------------------------------------------------
    #  Smart goal detection — ported from old stable pipeline
    # -----------------------------------------------------------------
    # A real goal sits in the (empty) net — the nearest player (the keeper) is
    # off the ball. A PASS ends with a receiver right on the ball. If an attacker
    # is this close to the ball when it's "in the goal region", it was caught,
    # not scored — this is what stopped normal passes near the wing/baseline
    # from being miscounted as goals.
    GOAL_RECEIVER_REJECT_PX = 62.0

    def _check_goal_smart(self, ball_track, tracks, frame_n: int, now: float) -> list:
        """Permissive goal-line check using a STATIC pixel region (12% of frame
        width on each side, middle 50% vertically). Triggered when:
            (a) ball is currently inside one of those goal regions, AND
            (b) ball history shows it was OUTSIDE the region 2-3 frames ago
                (i.e. it crossed in this attack), AND
            (c) we have evidence this was a real shot:
                - speed dropped sharply (ball hit the net), OR
                - speed was high (>15 px/f) recently, OR
                - ball just disappeared from inside the region, OR
                - a SHOT event fired in the last 3 seconds.
        This is the OLD stable logic that worked reliably on broadcast video."""
        if (frame_n - self._last_goal_frame) < GOAL_COOLDOWN_FRAMES:
            return []
        if len(self._ball_history) < 3:
            return []

        bx = ball_track.cx if ball_track else self._ball_history[-1][0]
        by = ball_track.cy if ball_track else self._ball_history[-1][1]

        in_left  = (bx < self._left_goal[2]) and \
                    (self._left_goal[1] < by < self._left_goal[3])
        in_right = (bx > self._right_goal[0]) and \
                    (self._right_goal[1] < by < self._right_goal[3])
        if not (in_left or in_right):
            return []

        # Evidence ball recently entered (was outside in last 2-3 ball positions)
        entered_recently = False
        for entry in self._ball_history[:-2]:
            hx = entry[0]
            if in_left and hx > self._left_goal[2]:
                entered_recently = True
            if in_right and hx < self._right_goal[0]:
                entered_recently = True
        if not entered_recently:
            return []

        # Pass-reception guard: if an attacker is right on the ball, it was
        # caught (a pass between players), not scored into the net. Only checked
        # when we actually have the ball's position this frame.
        if ball_track is not None and tracks:
            nearest_player = min(
                (math.hypot(bx - t.cx, by - t.cy)
                 for t in tracks if getattr(t, "class_id", 0) != 32),
                default=1e9)
            if nearest_player < self.GOAL_RECEIVER_REJECT_PX:
                return []

        # Speed evidence
        sp_now  = math.hypot(self._ball_history[-1][0] - self._ball_history[-2][0],
                              self._ball_history[-1][1] - self._ball_history[-2][1])
        sp_prev = math.hypot(self._ball_history[-2][0] - self._ball_history[-3][0],
                              self._ball_history[-2][1] - self._ball_history[-3][1])

        # A goal must FOLLOW A SHOT — not a pass. On broadcast footage the goal
        # posts are not visually detectable (the post-detector finds them in ~0%
        # of frames here), so the only thing guarding the static edge-strip is
        # this evidence chain. We require BOTH:
        #   (a) a real SHOT event in the last ~2.5s (a pass between players never
        #       registers as a shot toward the opponent goal), AND
        #   (b) "the ball reached the net": it decelerated sharply (hit the net)
        #       or vanished into it after travelling fast.
        # This is what finally stopped normal wing/baseline passes from being
        # miscounted as goals (the user's repeated complaint). The trade-off: a
        # goal where no shot was registered won't auto-count — flag it with the
        # A / B manual-goal keys.
        recent_shot = any(
            ev.event_type == EventType.SHOT and (now - ev.timestamp) < 2.5
            for ev in self.events[-40:])
        if not recent_shot:
            return []
        hit_net  = (sp_now < sp_prev * 0.6) and sp_prev > 8.0
        vanished = (ball_track is None) and sp_prev > 8.0
        if not (hit_net or vanished):
            return []

        # Attribution: which team scored, on which side
        side = "left" if in_left else "right"
        if self._a_attacks_right:
            team = "team_a" if in_right else "team_b"
        else:
            team = "team_a" if in_left else "team_b"

        self.score[team] += 1
        self._last_goal_frame = frame_n
        self._active_shot = None      # supersedes any pending shot
        return [MatchEvent(EventType.GOAL, team, now, frame_n, {
            "shooter_id":         self._current_player_id,
            "scored_by_player_id": self._current_player_id,
            "side":               side,
            "attribution_source": "goal_smart",
        })]

    # -----------------------------------------------------------------
    #  Direct goal detection — catches goals where SHOT was never fired
    # -----------------------------------------------------------------
    DIRECT_GOAL_HOLD_FRAMES = 4    # frames of real-detection ball inside goal
    DIRECT_GOAL_MIN_CONF    = 0.25 # reject low-conf / motion-only candidates

    def _check_direct_goal(self, ball_track, frame_n: int, now: float) -> list:
        """Fires a GOAL when a HIGH-CONFIDENCE real ball detection sits inside
        the dynamic goal box for ≥ DIRECT_GOAL_HOLD_FRAMES consecutive frames,
        even if no SHOT was previously registered. Strict to avoid noise:
        rejects phantom balls, motion-ball candidates, and weak YOLO candidates."""
        events = []
        gb = self._dynamic_goal_box
        if gb is None or ball_track is None:
            self._direct_goal_streak = 0
            return events
        # Cooldown: don't double-count
        if frame_n - self._last_goal_frame < GOAL_COOLDOWN_FRAMES:
            self._direct_goal_streak = 0
            return events

        # Require strong YOLO confidence — kicks out phantoms (~0.08-0.15)
        # and motion-only candidates (~0.30 synthetic). Real YOLO ball detections
        # near the goal post region typically score ≥ 0.30.
        ball_conf = float(getattr(ball_track, "confidence", 0.0))
        if ball_conf < self.DIRECT_GOAL_MIN_CONF:
            self._direct_goal_streak = 0
            return events

        bx, by = ball_track.cx, ball_track.cy
        try:
            pad = 10.0    # tighter than before — was 14
            inside_x = (gb.x_left - pad) <= bx <= (gb.x_right + pad)
            inside_y = (gb.y_top  - pad) <= by <= (gb.y_bottom + pad)
        except AttributeError:
            self._direct_goal_streak = 0
            return events

        if not (inside_x and inside_y):
            self._direct_goal_streak = 0
            return events

        self._direct_goal_streak = getattr(self, "_direct_goal_streak", 0) + 1
        if self._direct_goal_streak < self.DIRECT_GOAL_HOLD_FRAMES:
            return events

        # Confirmed: attribute the goal
        # Priority: current possession → last_passer → goal-side heuristic
        team = self.possession
        if team not in ("team_a", "team_b"):
            # Goal side heuristic: if goal box is on the right, team_a scored
            # (assuming team_a attacks right); otherwise team_b.
            cx_goal = (gb.x_left + gb.x_right) * 0.5
            on_right = cx_goal > self.fw / 2
            if on_right:
                team = "team_a" if self._a_attacks_right else "team_b"
            else:
                team = "team_b" if self._a_attacks_right else "team_a"

        self.score[team] += 1
        self._last_goal_frame = frame_n
        self._direct_goal_streak = 0
        # Cancel any pending shot since this goal supersedes it
        self._active_shot = None
        events.append(MatchEvent(EventType.GOAL, team, now, frame_n, {
            "shooter_id":             self._current_player_id,
            "scored_by_player_id":    self._current_player_id,
            "side":                   "right" if (gb.x_left + gb.x_right) * 0.5 > self.fw / 2 else "left",
            "attribution_source":     "direct_in_net",
        }))
        return events

    # FAST_BREAK cooldown: a fast break is ONE attack, not a per-frame event.
    # Without this the rule fires every infer frame for 10+ frames straight.
    FAST_BREAK_COOLDOWN_S    = 4.0
    FAST_BREAK_SPEED_MAX_PX  = 80.0   # phantom-jump cap

    def _check_fast_break(self, tracks, frame_n, now):
        events = []
        if not self.possession or len(self._ball_hist) < 3:
            return events

        # Cooldown: don't re-fire the same fast break every frame
        last_fb_ts = getattr(self, "_last_fast_break_ts", -1e9)
        if now - last_fb_ts < self.FAST_BREAK_COOLDOWN_S:
            return events

        # Possession-change recency: only count flips that happened far enough
        # back to be the START of an attack (≥ 0.4s) but still recent (< 3s).
        recent_change = any(
            ev.event_type == EventType.POSSESSION_CHANGE
            and 0.4 <= (now - ev.timestamp) < 3.0
            for ev in reversed(self.events[-10:])
        )
        if not recent_change:
            return events

        # Ball speed
        _, _, _, bf = self._ball_hist[-1]
        px, py, _, pf = self._ball_hist[-3]
        frames = max(bf - pf, 1)
        speed = math.hypot(self._ball_hist[-1][0] - px,
                            self._ball_hist[-1][1] - py) / frames
        # Reject phantom-jump speeds
        if speed < 14 or speed > self.FAST_BREAK_SPEED_MAX_PX:
            return events

        # Count defenders between ball and goal
        goal_x = self._get_attacking_goal_x(self.possession)
        defenders = 0
        for t in tracks:
            if getattr(t, 'team', '') == ("team_b" if self.possession == "team_a" else "team_a"):
                if (goal_x > self.fw / 2 and t.cx > self._ball_hist[-1][0]) or \
                   (goal_x < self.fw / 2 and t.cx < self._ball_hist[-1][0]):
                    defenders += 1
        # A real fast break needs at least SOME play visible — if defender count
        # is 0 it usually means we just don't see the defenders, not that they
        # are out of position. Require at least one tracked defender behind.
        if defenders < 1 or defenders > 2:
            return events

        self.fast_breaks[self.possession] += 1
        self._last_fast_break_ts = now
        events.append(MatchEvent(EventType.FAST_BREAK, self.possession, now, frame_n,
                                 {"defenders_back": defenders}))
        return events

    def _make_goal_regions(self):
        goal_w = int(self.fw * 0.12)
        goal_h = int(self.fh * 0.50)
        mid_y = self.fh // 2
        return ((0, mid_y - goal_h//2, goal_w, mid_y + goal_h//2),
                (self.fw - goal_w, mid_y - goal_h//2, self.fw, mid_y + goal_h//2))